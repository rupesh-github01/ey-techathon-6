import React, { useMemo, useState } from 'react';
import { Message, Visual } from '../hooks/useChat';
import VisualRenderer from './VisualRenderer';

const LIST_MARKER = /^(\d+[\.\)]|[-*•])\s+/;
const HEADER_LINE = /^[A-Z][A-Za-z0-9\s]{1,60}:?$/;

function replaceMetaLanguage(text: string) {
  const replacements = [
    { pattern: /\bcontexts?\b/gi, replacement: 'details' },
    { pattern: /\bagents?\b/gi, replacement: 'team' },
    { pattern: /\bdatasets?\b/gi, replacement: 'data' },
    { pattern: /\bsources?\b/gi, replacement: 'references' },
  ];
  return replacements.reduce((acc, { pattern, replacement }) => acc.replace(pattern, replacement), text);
}

// Remove/report-prefacing phrases to produce more direct statements
function normalizeTone(text: string) {
  return text.replace(/(^|\.|\n)\s*(Based on analysis[,\s]*)/gi, '$1')
    .replace(/(^|\.|\n)\s*(This reflects[,\s]*)/gi, '$1')
    .replace(/(^|\.|\n)\s*(The provided details[,\s]*)/gi, '$1');
}

function splitParagraphs(text: string, maxChars = 320) {
  const sentences = text.split(/(?<=[.!?])\s+/);
  if (sentences.length <= 1 && text.length <= maxChars) return [text.trim()];

  const paras: string[] = [];
  let current = '';

  const pushCurrent = () => {
    if (current.trim()) paras.push(current.trim());
    current = '';
  };

  (sentences.length > 1 ? sentences : text.split(/\s+/)).forEach((segment) => {
    const next = `${current} ${segment}`.trim();
    if (next.length > maxChars && current) {
      pushCurrent();
      current = segment;
    } else {
      current = next;
    }
  });

  pushCurrent();
  return paras;
}

function normalizeLines(section: string) {
  return section
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
}

function cleanListItem(line: string) {
  return line.replace(LIST_MARKER, '').trim();
}

// Collapse accidental spaced-letter tokens like "h e l l o" into "hello".
// Handles sequences separated by spaces or newlines and works when they appear
// inside longer blocks by replacing detected sequences with their collapsed form.
function collapseSpacedLetters(text: string) {
  if (!text) return text;
  const t = text.trim();

  // Fast path: whole-string match (common case). Allow zero-width and other unicode whitespace.
  const wholeMatch = t.match(/^([A-Za-z](?:[\s\u200B\u200C\u200D\uFEFF]+[A-Za-z]){2,})([\s\?\!\.,]*)$/);
  if (wholeMatch) {
    return wholeMatch[1].replace(/[\s\u200B\u200C\u200D\uFEFF]+/g, '') + (wholeMatch[2] || '');
  }

  // General replacement: find sequences of single letters separated by spaces/newlines or zero-width spaces
  return t.replace(/(?:\b[A-Za-z]\b(?:[\s\u200B\u200C\u200D\uFEFF]+))+[A-Za-z]\b/g, (m) => m.replace(/[\s\u200B\u200C\u200D\uFEFF]+/g, ''));
}

function formatContent(raw: string) {
  const sanitized = replaceMetaLanguage(raw || '');
  const normalized = sanitized.replace(/\r\n?/g, '\n').trim();
  if (!normalized) return null;

  const sections = normalized.split(/\n{2,}/).filter(Boolean);

  return sections
    .map((section, sectionIdx) => {
    const lines = normalizeLines(section);
    if (!lines.length) return null;

    let header: string | null = null;
    if (HEADER_LINE.test(lines[0])) {
      header = lines.shift() || null;
      if (header) header = header.replace(/:$/, '');
    }

    const listLikeCount = lines.filter((line) => LIST_MARKER.test(line)).length;
    const isList = listLikeCount >= 2 || (lines.length >= 3 && listLikeCount >= 1);

    if (isList) {
      const items = lines.map((line) => collapseSpacedLetters(cleanListItem(line)));
      return (
        <div className="message-section" key={`section-${sectionIdx}`}>
          {header && (
            <p className="message-section-header">
              <strong>{header}</strong>
            </p>
          )}
          <ul>
            {items.map((item, itemIdx) => (
              <li key={`li-${sectionIdx}-${itemIdx}`}>{item}</li>
            ))}
          </ul>
        </div>
      );
    }

    const paragraphText = lines.join(' ');
    const paragraphs = splitParagraphs(paragraphText);

    return (
      <div className="message-section" key={`section-${sectionIdx}`}>
        {header && (
          <p className="message-section-header">
            <strong>{header}</strong>
          </p>
        )}
        {paragraphs.map((para, paraIdx) => (
          <p key={`p-${sectionIdx}-${paraIdx}`}>{collapseSpacedLetters(para)}</p>
        ))}
      </div>
    );
  })
  .filter(Boolean);
}

export default function MessageBubble({ message }: { message: Message }) {
  const { role, content, visuals } = message;

  const formatted = useMemo(() => {
    if (role !== 'assistant') return null;
    return formatContent(normalizeTone(content));
  }, [content, role]);

  const [showFullContent, setShowFullContent] = useState(false);
  const [visibleVisualIndex, setVisibleVisualIndex] = useState<number | null>(null);

  const hasVisuals = Array.isArray(visuals) && visuals.length > 0;

  return (
    <div className={`message-wrapper ${role}`}>
      <div className="message-content">
        <div className="message-text">
          {role === 'assistant' && formatted ? (
            <div>
              {/* Show only a compact amount by default to keep chat concise */}
              {Array.isArray(formatted) ? (
                <div>
                  {(showFullContent ? formatted : formatted.slice(0, 8)).map((node: any, i: number) => (
                    <div key={i}>{node}</div>
                  ))}
                  {!showFullContent && (Array.isArray(formatted) && formatted.length > 8) && (
                    <div className="mt-2">
                      <button className="link-button" onClick={() => setShowFullContent(true)}>Want a deeper breakdown?</button>
                    </div>
                  )}
                </div>
              ) : (
                formatted
              )}
            </div>
          ) : collapseSpacedLetters(String(content || ''))}
        </div>

        {/* Render visuals (charts/tables) if present */}
        {hasVisuals && (
          <div className="message-visuals mt-4 space-y-4">
            {visuals!.map((visual: Visual, idx: number) => (
              <div key={idx} className="visual-container">
                {visual.title && (
                  <div className="visual-title text-sm font-semibold mb-2 text-white/80">
                    {visual.title}
                  </div>
                )}
                <VisualRenderer visual={visual} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
