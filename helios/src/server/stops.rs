use crate::{AdmissionError, Error, Result};

pub(super) struct TokenStops {
    sequences: Vec<Vec<u32>>,
    held: Vec<u32>,
}
impl TokenStops {
    pub fn new(sequences: Vec<Vec<u32>>, vocabulary: usize) -> Result<Self> {
        if sequences.len() > 16
            || sequences.iter().any(|sequence| {
                sequence.is_empty()
                    || sequence.len() > 64
                    || sequence.iter().any(|token| *token as usize >= vocabulary)
            })
        {
            return Err(Error::admission(
                AdmissionError::Invalid,
                "token stops allow at most sixteen nonempty sequences of at most 64 vocabulary IDs",
            ));
        }
        Ok(Self {
            sequences,
            held: Vec::with_capacity(64),
        })
    }
    pub fn push(&mut self, token: u32) -> (Vec<u32>, bool) {
        self.held.push(token);
        if let Some(length) = self
            .sequences
            .iter()
            .filter(|sequence| self.held.ends_with(sequence))
            .map(Vec::len)
            .max()
        {
            let visible = self.held.drain(..self.held.len() - length).collect();
            self.held.clear();
            return (visible, true);
        }
        let retain = self
            .sequences
            .iter()
            .flat_map(|sequence| {
                (1..sequence.len())
                    .rev()
                    .filter(|length| self.held.ends_with(&sequence[..*length]))
                    .take(1)
            })
            .max()
            .unwrap_or(0);
        (self.held.drain(..self.held.len() - retain).collect(), false)
    }
    pub fn finish(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.held)
    }
}

pub(super) fn validate_strings(strings: &[String], token_stops: bool, grammar: bool) -> Result<()> {
    if strings.len() > 4
        || strings.iter().any(String::is_empty)
        || strings.iter().map(String::len).sum::<usize>() > 4096
    {
        return Err(Error::admission(
            AdmissionError::Invalid,
            "stop allows at most four nonempty strings totaling at most 4096 UTF-8 bytes",
        ));
    }
    if !strings.is_empty() && token_stops {
        return Err(Error::admission(
            AdmissionError::Invalid,
            "stop cannot be combined with stop_token_ids or stop_token_sequences",
        ));
    }
    if !strings.is_empty() && grammar {
        return Err(Error::admission(
            AdmissionError::Invalid,
            "structured output cannot be combined with string stops",
        ));
    }
    Ok(())
}

pub(super) fn validate_initial_reasoning(
    syntax: Option<&crate::model::contract::ReasoningParser>,
) -> Result<()> {
    if syntax.is_some_and(|syntax| {
        syntax.kind != crate::model::contract::ReasoningParserKind::ThinkTags
    }) {
        return Err(Error::admission(
            AdmissionError::Feature,
            "after_reasoning string stops require text-delimited reasoning; token-delimited reasoning is unsupported",
        ));
    }
    Ok(())
}

struct StringStops {
    strings: Vec<String>,
    held: String,
    matched: Option<String>,
    initial_reasoning: Option<crate::model::contract::ReasoningParser>,
}
impl StringStops {
    fn new(strings: Vec<String>) -> Result<Self> {
        validate_strings(&strings, false, false)?;
        Ok(Self {
            strings,
            held: String::new(),
            matched: None,
            initial_reasoning: None,
        })
    }
    fn push(&mut self, text: &str) -> (String, bool) {
        if self.matched.is_some() {
            return (String::new(), true);
        }
        self.held.push_str(text);
        let mut reasoning_prefix = String::new();
        if let Some(syntax) = &self.initial_reasoning {
            // A close delimiter belongs to reasoning; an implicit terminal starts
            // the following envelope and must remain eligible for a user stop.
            let markers = std::iter::once((syntax.close.as_str(), true)).chain(
                syntax
                    .terminal_markers
                    .iter()
                    .map(|marker| (marker.as_str(), false)),
            );
            let boundary = markers
                .clone()
                .filter_map(|(marker, consume)| {
                    self.held
                        .find(marker)
                        .map(|at| (at, if consume { marker.len() } else { 0 }))
                })
                .min_by_key(|(at, _)| *at);
            if let Some((at, consume)) = boundary {
                reasoning_prefix = self.held.drain(..at + consume).collect();
                self.initial_reasoning = None;
            } else {
                let retain = markers
                    .flat_map(|(marker, _)| {
                        marker
                            .char_indices()
                            .skip(1)
                            .map(|(length, _)| length)
                            .filter(|length| self.held.ends_with(&marker[..*length]))
                    })
                    .max()
                    .unwrap_or(0);
                let visible = self.held.len() - retain;
                return (self.held.drain(..visible).collect(), false);
            }
        }
        if let Some((_, at, matched)) = self
            .strings
            .iter()
            .enumerate()
            .filter_map(|(i, stop)| self.held.find(stop).map(|at| (at + stop.len(), at, i)))
            .min()
        {
            let visible = self.held[..at].to_owned();
            self.held.clear();
            self.matched = Some(self.strings[matched].clone());
            let visible = if reasoning_prefix.is_empty() {
                visible
            } else {
                reasoning_prefix + &visible
            };
            return (visible, true);
        }
        let retain = self
            .strings
            .iter()
            .flat_map(|stop| {
                stop.char_indices()
                    .skip(1)
                    .map(|(length, _)| length)
                    .filter(|length| self.held.ends_with(&stop[..*length]))
            })
            .max()
            .unwrap_or(0);
        let visible: String = self.held.drain(..self.held.len() - retain).collect();
        let visible = if reasoning_prefix.is_empty() {
            visible
        } else {
            reasoning_prefix + &visible
        };
        (visible, false)
    }
    fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

struct Span {
    tokens: Vec<u32>,
    text: String,
    entries: Vec<(usize, super::token_spans::ReturnedLogprob)>,
}

pub(super) struct TextStops<'a> {
    decoder: crate::frontend::IncrementalDecoder<'a>,
    tracked: Option<super::token_spans::TokenSpans<'a>>,
    pending_entries: Vec<(usize, super::token_spans::ReturnedLogprob)>,
    stops: StringStops,
    pending_tokens: Vec<u32>,
    spans: std::collections::VecDeque<Span>,
    visible_bytes: usize,
}
impl<'a> TextStops<'a> {
    pub fn matched(&self) -> Option<&str> {
        self.stops.matched.as_deref()
    }
    pub fn new(frontend: &'a crate::frontend::Frontend, strings: Vec<String>) -> Result<Self> {
        Ok(Self {
            decoder: frontend.incremental_decoder(false),
            tracked: None,
            pending_entries: vec![],
            stops: StringStops::new(strings)?,
            pending_tokens: Vec::new(),
            spans: std::collections::VecDeque::new(),
            visible_bytes: 0,
        })
    }
    pub fn with_initial_reasoning(
        mut self,
        syntax: Option<&crate::model::contract::ReasoningParser>,
    ) -> Result<Self> {
        validate_initial_reasoning(syntax)?;
        self.stops.initial_reasoning = syntax.cloned();
        Ok(self)
    }
    pub fn track(&mut self, frontend: &'a crate::frontend::Frontend) {
        self.tracked = Some(super::token_spans::TokenSpans::new(frontend));
    }
    pub fn push_logprob(
        &mut self,
        token: u32,
        value: crate::execution::TokenLogprobs,
        silent: bool,
    ) -> Result<(Vec<crate::output::Piece>, bool)> {
        let tracked = self.tracked.as_mut().ok_or("logprob stop decoder was not initialized")?;
        if silent {
            return Ok(match tracked.silent(token, Some(value)) {
                Some(span) => self.accept_span(span, false),
                None => (vec![], false),
            });
        }
        match tracked.push(token, Some(value))? {
            Some(span) => Ok(self.accept_span(span, false)),
            None => Ok((vec![], false)),
        }
    }
    fn accept_span(&mut self, span: super::token_spans::Span, finish: bool) -> (Vec<crate::output::Piece>, bool) {
        let silent = !span.tokens.is_empty()
            && span.tokens.iter().all(|token| token.logprob.as_ref().is_some_and(|e| e.silent));
        for token in span.tokens {
            self.pending_tokens.push(token.token);
            if let Some(entry) = token.logprob {
                self.pending_entries.push((token.start, entry));
            }
        }
        if silent && span.text.is_empty() {
            self.spans.push_back(Span {
                text: String::new(), tokens: std::mem::take(&mut self.pending_tokens),
                entries: std::mem::take(&mut self.pending_entries),
            });
        }
        self.accept_text(span.text, finish)
    }
    pub fn push(&mut self, token: u32) -> Result<(Vec<crate::output::Piece>, bool)> {
        self.pending_tokens.push(token);
        let text = self.decoder.push(&[token])?;
        Ok(self.accept_text(text.unwrap_or_default(), false))
    }
    pub fn finish(&mut self, eos: Option<u32>) -> Result<(Vec<crate::output::Piece>, bool)> {
        let (mut pieces, matched) = if let Some(tracked) = &mut self.tracked {
            let span = tracked.finish()?;
            self.accept_span(span, true)
        } else {
            let text = self.decoder.finish()?;
            self.accept_text(text, true)
        };
        if !matched {
            pieces.extend(self.pending_entries.drain(..).map(|(_, entry)| crate::output::Piece::Logprob(entry)));
            pieces.extend(
                self.pending_tokens
                    .drain(..)
                    .map(crate::output::Piece::Token),
            );
            pieces.extend(eos.map(crate::output::Piece::Token));
        }
        Ok((pieces, matched))
    }
    fn accept_text(&mut self, text: String, finish: bool) -> (Vec<crate::output::Piece>, bool) {
        use crate::output::Piece;
        let (visible, matched) = self.stops.push(&text);
        self.visible_bytes += visible.len();
        if !text.is_empty() {
            self.spans.push_back(Span {
                tokens: std::mem::take(&mut self.pending_tokens),
                entries: std::mem::take(&mut self.pending_entries),
                text,
            });
        }
        if finish && !matched {
            self.visible_bytes += self.stops.finish().len();
        }
        let mut pieces = Vec::new();
        while self
            .spans
            .front()
            .is_some_and(|span| span.text.len() <= self.visible_bytes)
        {
            let span = self.spans.pop_front().unwrap();
            self.visible_bytes -= span.text.len();
            pieces.extend(span.entries.into_iter().map(|(_, entry)| Piece::Logprob(entry)));
            pieces.extend(span.tokens.into_iter().map(Piece::Token));
            pieces.push(Piece::Text(span.text));
        }
        if matched {
            // A clipped token cannot act as a whole-token protocol control.
            if let Some(span) = self.spans.front() {
                let mut discarded = false;
                for (start, entry) in &span.entries {
                    if *start < self.visible_bytes || (entry.silent && !discarded) {
                        pieces.push(Piece::Logprob(entry.clone()));
                    } else {
                        discarded = true;
                    }
                }
                if self.visible_bytes != 0 {
                    pieces.push(Piece::Text(span.text[..self.visible_bytes].to_owned()));
                }
            }
            self.visible_bytes = 0;
            self.pending_tokens.clear();
            self.pending_entries.clear();
            self.spans.clear();
        }
        (pieces, matched)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_stop_admission_has_typed_errors() {
        assert!(matches!(
            TokenStops::new(vec![vec![]], 10).err().unwrap(),
            Error::Admission(AdmissionError::Invalid(_))
        ));
        assert!(matches!(
            validate_strings(&[String::new()], false, false).unwrap_err(),
            Error::Admission(AdmissionError::Invalid(_))
        ));
        assert!(matches!(
            validate_strings(&["end".into()], true, false).unwrap_err(),
            Error::Admission(AdmissionError::Invalid(_))
        ));
    }

    #[test]
    fn overlapping_stop_prefixes_are_held_without_losing_ordinary_tokens() {
        let mut stops = TokenStops::new(vec![vec![1, 2, 3], vec![2, 4]], 248320).unwrap();
        assert_eq!(stops.push(9), (vec![9], false));
        assert_eq!(stops.push(1), (vec![], false));
        assert_eq!(stops.push(2), (vec![], false));
        assert_eq!(stops.push(4), (vec![1], true));
        assert!(stops.finish().is_empty());
    }
    #[test]
    fn terminal_flush_preserves_an_unfinished_stop_prefix() {
        let mut stops = TokenStops::new(vec![vec![1, 2, 3]], 248320).unwrap();
        assert_eq!(stops.push(1), (vec![], false));
        assert_eq!(stops.finish(), vec![1]);
    }
}

#[cfg(test)]
pub(super) mod string_tests {
    use super::*;
    use crate::frontend::{Asset, Frontend, FrontendAssets};
    use sha2::{Digest, Sha256};

    pub(crate) fn frontend() -> Frontend {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "flare-stop-tokenizer-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let vocab: tokenizers::models::bpe::Vocab = [
            "before", "ENDafter", "<0xE4>", "<0xB8>", "<0x96>", "<0xE7>", "<0x95>", "<0x8C>", "x",
            "ST", "OP", "<0xEF>", "<0xBF>", "<0xBD>",
        ]
        .into_iter()
        .enumerate()
        .map(|(i, t)| (t.to_owned(), i as u32))
        .collect();
        let model = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_decoder(Some(tokenizers::decoders::sequence::Sequence::new(vec![
            tokenizers::decoders::byte_fallback::ByteFallback::default().into(),
            tokenizers::decoders::fuse::Fuse::default().into(),
        ])));
        let asset = |name: &str, bytes: &[u8]| {
            let path = root.join(name);
            std::fs::write(&path, bytes).unwrap();
            Asset {
                range: None,
                path,
                sha256: format!("{:x}", Sha256::digest(bytes)),
            }
        };
        let result = Frontend::load(
            FrontendAssets {
                tokenizer: asset(
                    "tokenizer.json",
                    tokenizer.to_string(false).unwrap().as_bytes(),
                ),
                tokens: Default::default(),
                chat_template: Some(asset("chat_template.jinja", b"{{ messages[0].content }}")),
            },
            Default::default(),
        )
        .unwrap();
        std::fs::remove_dir_all(root).unwrap();
        result
    }

    fn reasoning(
        close: &str,
        terminal_markers: &[&str],
    ) -> crate::model::contract::ReasoningParser {
        let xtml = close == "<|close|>think<|sep|>";
        crate::model::contract::ReasoningParser {
            kind: if xtml {
                crate::model::contract::ReasoningParserKind::XtmlThink
            } else {
                crate::model::contract::ReasoningParserKind::ThinkTags
            },
            open: if xtml {
                "<|open|>think<|sep|>"
            } else {
                "<think>"
            }
            .into(),
            close: close.into(),
            wrapping_newlines: 0,
            terminal_markers: terminal_markers.iter().map(|s| (*s).into()).collect(),
        }
    }

    fn xtml_frontend() -> Frontend {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "flare-stop-xtml-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&root).unwrap();
        let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
            .into_iter()
            .collect();
        alphabet.sort_unstable();
        let vocabulary: tokenizers::models::bpe::Vocab = alphabet
            .into_iter()
            .enumerate()
            .map(|(id, character)| (character.to_string(), id as u32))
            .collect();
        let mut tokenizer = tokenizers::Tokenizer::new(
            tokenizers::models::bpe::BPE::builder()
                .vocab_and_merges(vocabulary, vec![])
                .build()
                .unwrap(),
        );
        tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        tokenizer
            .add_special_tokens(
                ["<|open|>", "<|close|>", "<|sep|>", "<eos>"]
                    .map(|marker| tokenizers::AddedToken::from(marker, true)),
            )
            .unwrap();
        let asset = |name: &str, bytes: &[u8]| {
            let path = root.join(name);
            std::fs::write(&path, bytes).unwrap();
            Asset {
                path,
                range: None,
                sha256: format!("{:x}", Sha256::digest(bytes)),
            }
        };
        let frontend = Frontend::load(FrontendAssets {
            tokenizer: asset("tokenizer.json", tokenizer.to_string(false).unwrap().as_bytes()),
            tokens: Default::default(),
            chat_template: Some(asset("chat.jinja", b"{% for message in messages %}{{ message.content }}{% endfor %}<|assistant|><|open|>think<|sep|>")),
        }, serde_json::json!({"add_generation_prompt":true}).as_object().unwrap().clone()).unwrap();
        std::fs::remove_dir_all(root).unwrap();
        frontend
    }

    fn xtml_output(frontend: &Frontend) -> crate::output::OutputConfig {
        use crate::model::contract::{ToolParser, ToolParserKind};
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.assistant_marker = "<|assistant|>".into();
        syntax.reasoning = Some(reasoning("<|close|>think<|sep|>", &[]));
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::Xtml,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        crate::output::OutputConfig {
            syntax,
            content: crate::output::tagged::OutputConfig {
                structured: false,
                start: crate::output::tagged::OutputStart::Reasoning,
                tools: vec![],
            },
            control_tokens: ["<|open|>", "<|close|>", "<|sep|>"]
                .into_iter()
                .map(|marker| {
                    let ids = frontend.encode(marker, false).unwrap();
                    assert_eq!(ids.len(), 1);
                    (ids[0], marker.into())
                })
                .collect(),
        }
    }

    #[test]
    fn xtml_literal_close_and_real_control_close_keep_distinct_parser_semantics() {
        use crate::output::{OutputPipeline, Termination};
        use crate::server::output::{Accumulator, Decoder};
        let frontend = xtml_frontend();
        let config = xtml_output(&frontend);
        let literal = "<|close|>think<|sep|>STOP";
        let ordinary: Vec<u32> = literal
            .chars()
            .map(|character| {
                frontend
                    .tokenizer()
                    .token_to_id(&character.to_string())
                    .unwrap()
            })
            .collect();
        let controls = frontend
            .encode(
                "reason<|close|>think<|sep|><|open|>response<|sep|>answer",
                false,
            )
            .unwrap();
        for (tokens, expected_reason, expected_answer) in [
            (ordinary.clone(), Some(literal), None),
            (controls, Some("reason"), Some("answer")),
        ] {
            let mut decoder = Decoder::<OutputPipeline>::new(&frontend, &[], Some(config.clone()));
            let mut output = Accumulator::new("xtml".into());
            for token in tokens {
                output.apply(decoder.push(token).unwrap()).unwrap();
            }
            output
                .apply(decoder.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            assert_eq!(
                output.message()["reasoning_content"].as_str(),
                expected_reason
            );
            assert_eq!(output.message()["content"].as_str(), expected_answer);
        }
        let syntax = config.syntax.reasoning.as_ref().unwrap();
        assert!(matches!(
            TextStops::new(&frontend, vec!["STOP".into()])
                .unwrap()
                .with_initial_reasoning(Some(syntax))
                .err()
                .unwrap(),
            Error::Admission(AdmissionError::Feature(_)),
        ));
        let mut legacy = TextStops::new(&frontend, vec!["STOP".into()]).unwrap();
        let mut visible = String::new();
        let mut matched = false;
        for token in ordinary {
            let (pieces, hit) = legacy.push(token).unwrap();
            visible.push_str(&text(pieces));
            matched |= hit;
        }
        assert!(matched);
        assert_eq!(visible, "<|close|>think<|sep|>");
    }

    #[test]
    fn initial_xtml_after_reasoning_scope_is_rejected_before_request_admission() {
        let frontend = xtml_frontend();
        let output = xtml_output(&frontend);
        let mut config =
            crate::frontend::test_support::model_config(crate::model::testing::metadata());
        config.contract.frontend = output.syntax;
        config.contract.vocabulary = frontend.tokenizer().get_vocab_size(true);
        config.contract.input_vocabulary = config.contract.vocabulary;
        config.eos_token_ids = vec![frontend.tokenizer().token_to_id("<eos>").unwrap()];
        let compiler = crate::grammar::GrammarCompiler::new(
            frontend.tokenizer(),
            config.contract.vocabulary,
            &config.eos_token_ids,
        )
        .unwrap();
        for scope in [None, Some("all"), Some("after_reasoning")] {
            let mut body = serde_json::json!({
                "model":config.model_id, "messages":[{"role":"user","content":"hello"}],
                "stop":"STOP", "max_tokens":8,
            });
            if let Some(scope) = scope {
                body["stop_scope"] = scope.into();
            }
            let prepared =
                crate::server::request::prepare(&frontend, None, &compiler, &config, body);
            if scope == Some("after_reasoning") {
                assert!(matches!(
                    prepared.err().unwrap(),
                    Error::Admission(AdmissionError::Feature(_))
                ));
            } else {
                let prepared = prepared.unwrap();
                assert_eq!(
                    prepared.output.content.start,
                    crate::output::tagged::OutputStart::Reasoning
                );
                assert!(prepared.prompt.string_stop_reasoning.is_none());
            }
        }
    }

    #[test]
    fn initial_reasoning_stops_are_independent_of_marker_and_stop_chunk_boundaries() {
        let input = "<think>Question: STOP</think>answerSTOPdiscard";
        for first in 0..=input.len() {
            for second in first..=input.len() {
                let mut stops =
                    StringStops::new(vec!["Question:".into(), "STOP".into(), "</th".into()])
                        .unwrap();
                stops.initial_reasoning = Some(reasoning("</think>", &[]));
                let mut visible = String::new();
                let mut matched = false;
                for chunk in [&input[..first], &input[first..second], &input[second..]] {
                    let (text, hit) = stops.push(chunk);
                    visible.push_str(&text);
                    matched |= hit;
                }
                assert!(matched, "splits {first}/{second}");
                assert_eq!(
                    visible, "<think>Question: STOP</think>answer",
                    "splits {first}/{second}"
                );
                assert_eq!(stops.matched.as_deref(), Some("STOP"));
            }
        }
    }

    #[test]
    fn reasoning_guard_uses_declared_close_and_tool_terminal_without_matching_across_them() {
        let mut stops = StringStops::new(vec!["STOP".into()]).unwrap();
        stops.initial_reasoning = Some(reasoning("</think>", &[]));
        assert_eq!(stops.push("ST</thi"), ("ST".into(), false));
        assert_eq!(stops.push("nk>OP"), ("</think>OP".into(), false));
        assert_eq!(stops.push("<think>STOP"), ("<think>".into(), true));

        let mut stops = StringStops::new(vec!["STOP".into(), "<tool_call>".into()]).unwrap();
        stops.initial_reasoning = Some(reasoning("</think>", &["<tool_call>"]));
        assert_eq!(stops.push("STOP<tool_"), ("STOP".into(), false));
        assert_eq!(stops.push("call>discard"), (String::new(), true));
        assert_eq!(stops.matched.as_deref(), Some("<tool_call>"));
    }

    #[test]
    fn decoder_clips_answer_inside_the_same_token_that_closes_reasoning() {
        use crate::output::Piece;
        let frontend = crate::server::output::tests::frontend(&["STOP</think>answerSTOPdiscard"]);
        let syntax = reasoning("</think>", &[]);
        let mut stops = TextStops::new(&frontend, vec!["STOP".into()])
            .unwrap()
            .with_initial_reasoning(Some(&syntax))
            .unwrap();
        assert_eq!(
            stops.push(0).unwrap(),
            (vec![Piece::Text("STOP</think>answer".into())], true)
        );
        assert_eq!(stops.finish(Some(99)).unwrap(), (vec![], true));
        let mut all = TextStops::new(&frontend, vec!["STOP".into()]).unwrap();
        assert_eq!(all.push(0).unwrap(), (vec![], true));
    }

    #[test]
    fn ordinary_scope_keeps_stops_inside_fragmented_reasoning_markup() {
        let frontend =
            crate::server::output::tests::frontend(&["<thi", "nk>reasonST", "OP</think>answer"]);
        let mut stops = TextStops::new(&frontend, vec!["STOP".into()]).unwrap();
        let mut visible = String::new();
        for token in [0, 1, 2] {
            let (text, matched) = push_text(&mut stops, token);
            visible.push_str(&text);
            assert_eq!(matched, token == 2);
        }
        assert_eq!(visible, "<think>reason");
        assert_eq!(finish_text(&mut stops), (String::new(), true));
    }

    #[test]
    fn reasoning_guard_preserves_utf8_and_flushes_unclosed_reasoning_at_normal_termination() {
        let frontend = frontend();
        let syntax = reasoning("</think>", &[]);
        let mut stops = TextStops::new(&frontend, vec!["世界".into()])
            .unwrap()
            .with_initial_reasoning(Some(&syntax))
            .unwrap();
        let mut visible = String::new();
        for token in [2, 3, 4, 5, 6, 7] {
            let (text, matched) = push_text(&mut stops, token);
            visible.push_str(&text);
            assert!(!matched);
        }
        assert_eq!(visible, "世界");
        assert_eq!(finish_text(&mut stops), (String::new(), false));

        let frontend = crate::server::output::tests::frontend(&["STOP</thi"]);
        for eos in [None, Some(99)] {
            let mut stops = TextStops::new(&frontend, vec!["STOP".into(), "</th".into()])
                .unwrap()
                .with_initial_reasoning(Some(&syntax))
                .unwrap();
            let (mut pieces, matched) = stops.push(0).unwrap();
            assert!(!matched);
            let (tail, matched) = stops.finish(eos).unwrap();
            assert!(!matched);
            pieces.extend(tail);
            assert_eq!(text(pieces), "STOP</thi");
        }
    }

    #[test]
    fn reasoning_guard_and_output_pipeline_keep_reasoning_and_answer_separate() {
        use crate::output::{
            Input, OutputConfig, OutputPipeline, Piece, Termination,
            tagged::{OutputConfig as TaggedConfig, OutputStart},
        };
        use crate::server::output::Accumulator;
        let pieces = ["Question:ST", "OP</thi", "nk>answerST", "OPdiscard"];
        let frontend = crate::server::output::tests::frontend(&pieces);
        let syntax = reasoning("</think>", &[]);
        for cause in [
            Termination::UserStop,
            Termination::Eos,
            Termination::OutputLimit,
        ] {
            let mut contract = crate::model::testing::metadata().frontend;
            contract.reasoning = Some(syntax.clone());
            let mut parser = OutputPipeline::new(OutputConfig {
                syntax: contract,
                control_tokens: vec![],
                content: TaggedConfig {
                    start: OutputStart::Reasoning,
                    structured: false,
                    tools: vec![],
                },
            });
            let mut stops = TextStops::new(&frontend, vec!["Question:".into(), "STOP".into()])
                .unwrap()
                .with_initial_reasoning(Some(&syntax))
                .unwrap();
            let mut output = Accumulator::new("guard".into());
            let mut last_match = false;
            for token in 0..if cause == Termination::UserStop { 4 } else { 3 } {
                let (pieces, matched) = stops.push(token).unwrap();
                last_match = matched;
                for piece in pieces {
                    let events = match piece {
                        Piece::Token(token) => parser.push(Input::Token(token)).unwrap(),
                        Piece::Text(text) => parser.push(Input::Text(&text)).unwrap(),
                        Piece::Logprob(_) => unreachable!(),
                    };
                    output.apply(events).unwrap();
                }
            }
            assert_eq!(last_match, cause == Termination::UserStop);
            let (tail, matched) = stops.finish(None).unwrap();
            assert_eq!(matched, cause == Termination::UserStop);
            for piece in tail {
                if let Piece::Text(text) = piece {
                    output
                        .apply(parser.push(Input::Text(&text)).unwrap())
                        .unwrap();
                }
            }
            output.apply(parser.finish(cause).unwrap()).unwrap();
            output.finish().unwrap();
            assert_eq!(output.message()["reasoning_content"], "Question:STOP");
            assert_eq!(
                output.message()["content"],
                if cause == Termination::UserStop {
                    "answer"
                } else {
                    "answerST"
                }
            );
        }
    }

    #[test]
    fn cancelling_or_failing_does_not_flush_a_pending_reasoning_close_marker() {
        use crate::output::{
            Input, OutputConfig, OutputPipeline, Piece, Termination,
            tagged::{OutputConfig as TaggedConfig, OutputStart},
        };
        use crate::server::output::Accumulator;
        let frontend = crate::server::output::tests::frontend(&["reasonSTOP", "</thi"]);
        let syntax = reasoning("</think>", &[]);
        for cause in [Termination::Cancelled, Termination::ExecutionError] {
            let mut contract = crate::model::testing::metadata().frontend;
            contract.reasoning = Some(syntax.clone());
            let mut parser = OutputPipeline::new(OutputConfig {
                syntax: contract,
                control_tokens: vec![],
                content: TaggedConfig {
                    start: OutputStart::Reasoning,
                    structured: false,
                    tools: vec![],
                },
            });
            let mut output = Accumulator::new("cancel".into());
            let mut stops = TextStops::new(&frontend, vec!["STOP".into()])
                .unwrap()
                .with_initial_reasoning(Some(&syntax))
                .unwrap();
            for token in [0, 1] {
                let (pieces, matched) = stops.push(token).unwrap();
                assert!(!matched);
                for piece in pieces {
                    if let Piece::Text(text) = piece {
                        output
                            .apply(parser.push(Input::Text(&text)).unwrap())
                            .unwrap();
                    }
                }
            }
            assert_eq!(output.message()["reasoning_content"], "reasonSTOP");
            // Cancellation drops the worker's filter without the EOS/length flush.
            drop(stops);
            assert!(parser.finish(cause).unwrap().is_empty());
            assert!(!output.message().to_string().contains("</thi"));
        }
    }

    #[test]
    fn first_stop_wins_inside_a_token_and_overlapping_prefixes_are_held() {
        for splits in [vec!["abcSTOPtailEND"], vec!["a", "bcST", "OPtailEND"]] {
            let mut stops =
                StringStops::new(vec!["END".into(), "STOP".into(), "STORK".into()]).unwrap();
            let mut visible = String::new();
            let mut matched = false;
            for piece in splits {
                let (text, hit) = stops.push(piece);
                visible.push_str(&text);
                matched |= hit;
            }
            assert_eq!(visible, "abc");
            assert!(matched);
            assert_eq!(stops.matched.as_deref(), Some("STOP"));
            assert_eq!(stops.push("more"), (String::new(), true));
            assert_eq!(stops.finish(), "");
        }
        let mut stops = StringStops::new(vec!["abc".into(), "bX".into()]).unwrap();
        assert_eq!(stops.push("ab"), (String::new(), false));
        assert_eq!(stops.push("Xafter"), ("a".into(), true));
    }

    #[test]
    fn every_utf8_chunk_boundary_preserves_visible_prefix_and_unmatched_suffix() {
        let full = "café 世界ENDdiscard";
        for split in full.char_indices().map(|(at, _)| at) {
            let mut stops = StringStops::new(vec!["世界END".into(), "é!".into()]).unwrap();
            let (a, _) = stops.push(&full[..split]);
            let (b, hit) = stops.push(&full[split..]);
            assert!(hit);
            assert_eq!(a + &b, "café ");
            assert_eq!(stops.matched.as_deref(), Some("世界END"));
        }
        let mut stops = StringStops::new(vec!["世界END".into()]).unwrap();
        assert_eq!(stops.push("ok世界EN"), ("ok".into(), false));
        assert_eq!(stops.finish(), "世界EN");
    }

    fn text(pieces: Vec<crate::output::Piece>) -> String {
        pieces
            .into_iter()
            .filter_map(|piece| match piece {
                crate::output::Piece::Text(text) => Some(text),
                _ => None,
            })
            .collect()
    }
    fn push_text(stops: &mut TextStops<'_>, token: u32) -> (String, bool) {
        let (pieces, matched) = stops.push(token).unwrap();
        (text(pieces), matched)
    }
    fn finish_text(stops: &mut TextStops<'_>) -> (String, bool) {
        let (pieces, matched) = stops.finish(None).unwrap();
        (text(pieces), matched)
    }

    #[test]
    fn incremental_decoder_handles_split_utf8_and_subtoken_stops() {
        let frontend = frontend();
        let mut stops = TextStops::new(&frontend, vec!["世界".into()]).unwrap();
        assert_eq!(push_text(&mut stops, 0), ("before".into(), false));
        for token in [2, 3, 4, 5, 6] {
            assert_eq!(push_text(&mut stops, token), (String::new(), false));
        }
        assert_eq!(push_text(&mut stops, 7), (String::new(), true));
        assert_eq!(finish_text(&mut stops), (String::new(), true));
        let mut stops = TextStops::new(&frontend, vec!["END".into()]).unwrap();
        assert_eq!(push_text(&mut stops, 0), ("before".into(), false));
        assert_eq!(push_text(&mut stops, 1), (String::new(), true));
        let mut stops = TextStops::new(&frontend, vec!["STOP".into()]).unwrap();
        assert_eq!(push_text(&mut stops, 9), (String::new(), false));
        assert_eq!(finish_text(&mut stops), ("ST".into(), false));
        let mut stops = TextStops::new(&frontend, vec!["\u{fffd}".into()]).unwrap();
        assert_eq!(push_text(&mut stops, 2), (String::new(), false));
        assert_eq!(finish_text(&mut stops), (String::new(), true));
    }

    #[test]
    fn invalid_or_incompatible_stop_contracts_are_rejected() {
        for values in [
            vec![String::new()],
            vec!["a".into(); 5],
            vec!["x".repeat(4097)],
        ] {
            assert!(validate_strings(&values, false, false).is_err());
        }
        assert!(validate_strings(&["STOP".into()], true, false).is_err());
        assert!(validate_strings(&["STOP".into()], false, true).is_err());
        assert!(validate_strings(&[], true, true).is_ok());
    }
    #[test]
    fn held_spans_release_before_later_token_controls_and_clipped_tokens_have_no_control() {
        use crate::output::Piece::{Text, Token};
        let frontend =
            crate::server::output::tests::frontend(&["beforeST", "<control>", "visibleENDhidden"]);
        let mut stops = TextStops::new(&frontend, vec!["STOP".into()]).unwrap();
        assert_eq!(stops.push(0).unwrap(), (vec![], false));
        assert_eq!(
            stops.push(1).unwrap(),
            (
                vec![
                    Token(0),
                    Text("beforeST".into()),
                    Token(1),
                    Text("<control>".into())
                ],
                false
            )
        );
        assert_eq!(stops.finish(Some(99)).unwrap(), (vec![Token(99)], false));
        let mut stops = TextStops::new(&frontend, vec!["END".into()]).unwrap();
        assert_eq!(stops.push(2).unwrap(), (vec![Text("visible".into())], true));
        assert_eq!(stops.finish(Some(99)).unwrap(), (vec![], true));
    }
    #[test]
    fn first_completed_overlap_is_independent_of_every_chunk_partition() {
        let input = "abcdef";
        for split_bits in 0..(1 << (input.len() - 1)) {
            let mut stops = StringStops::new(vec!["abcdef".into(), "bc".into()]).unwrap();
            let mut start = 0;
            let mut output = String::new();
            let mut matched = false;
            for end in 1..=input.len() {
                if end == input.len() || split_bits & (1 << (end - 1)) != 0 {
                    let (text, hit) = stops.push(&input[start..end]);
                    output.push_str(&text);
                    matched |= hit;
                    start = end;
                }
            }
            assert!(matched);
            assert_eq!(output, "a", "partition {split_bits}");
        }
        let mut stops = StringStops::new(vec!["bc".into(), "abc".into()]).unwrap();
        assert_eq!(stops.push("abc"), (String::new(), true));
    }
}
