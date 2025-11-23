'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
}

interface MessageListProps {
  messages: Message[];
  loading: boolean;
}

export default function MessageList({ messages, loading }: MessageListProps) {
  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-4">
      {messages.length === 0 && !loading && (
        <div className="text-center text-gray-500 mt-20">
          <svg 
            className="w-16 h-16 mx-auto mb-4 text-gray-400" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" 
            />
          </svg>
          <p className="text-lg font-semibold">Start a conversation</p>
          <p className="text-sm">Ask anything about your company documents</p>
        </div>
      )}

      {messages.map((message, index) => (
        <div
          key={index}
          className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
        >
          <div
            className={`max-w-3xl rounded-lg p-4 ${
              message.role === 'user'
                ? 'bg-blue-600 text-white'
                : 'bg-white border border-gray-200'
            }`}
          >
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              className={`prose ${message.role === 'user' ? 'prose-invert' : ''} max-w-none`}
            >
              {message.content}
            </ReactMarkdown>
            
            {message.sources && message.sources.length > 0 && (
              <div className="mt-3 pt-3 border-t border-gray-200">
                <p className="text-xs text-gray-600 font-semibold mb-1">Sources:</p>
                <ul className="text-xs text-gray-600 space-y-1">
                  {message.sources.map((source, idx) => (
                    <li key={idx}>• {source}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      ))}

      {loading && (
        <div className="flex justify-start">
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="flex space-x-2">
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
              <div 
                className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" 
                style={{ animationDelay: '0.2s' }}
              ></div>
              <div 
                className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" 
                style={{ animationDelay: '0.4s' }}
              ></div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}