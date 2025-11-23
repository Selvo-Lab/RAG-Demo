'use client';

import { useRef } from 'react';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

interface SidebarProps {
  documentCount: number;
  onDocumentCountChange: () => void;
  onClearMessages: () => void;
}

export default function Sidebar({ 
  documentCount, 
  onDocumentCountChange,
  onClearMessages 
}: SidebarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post(`${API_URL}/api/documents/upload-file`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      onDocumentCountChange();
      alert('File uploaded successfully!');
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Failed to upload file');
    }
  };

  const handleClearDocuments = async () => {
    if (!window.confirm('Are you sure you want to clear all documents?')) return;

    try {
      await axios.delete(`${API_URL}/api/documents/clear`);
      onClearMessages();
      onDocumentCountChange();
      alert('All documents cleared!');
    } catch (error) {
      console.error('Error clearing documents:', error);
      alert('Failed to clear documents');
    }
  };

  return (
    <div className="w-80 bg-white border-r border-gray-200 p-6 flex flex-col h-screen">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Knowledge Base</h2>
      
      <div className="mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Documents in system</p>
          <p className="text-3xl font-bold text-blue-600">{documentCount}</p>
        </div>
      </div>

      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Upload Document
        </label>
        <input
          ref={fileInputRef}
          type="file"
          onChange={handleFileUpload}
          accept=".pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg"
          className="w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
        <p className="text-xs text-gray-500 mt-2">
          Supports: PDF, Word, PowerPoint, Text, Images
        </p>
      </div>

      <button
        onClick={handleClearDocuments}
        className="mt-auto bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition"
      >
        Clear All Documents
      </button>

      <div className="mt-4 text-xs text-gray-500 text-center">
        Company Knowledge Assistant v2.0
      </div>
    </div>
  );
}