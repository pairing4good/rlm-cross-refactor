'use client';

import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';
import { RLMIteration, extractFinalAnswer } from '@/lib/types';

interface TrajectoryPanelProps {
  iterations: RLMIteration[];
  selectedIteration: number;
  onSelectIteration: (index: number) => void;
}

// Helper to format message content for display
function formatMessageContent(content: string): string {
  // Truncate very long content
  if (content.length > 3000) {
    return content.slice(0, 3000) + '\n\n... [content truncated for display]';
  }
  return content;
}

// Render a role badge
function RoleBadge({ role }: { role: string }) {
  const config = {
    system: { label: 'System', className: 'bg-violet-500/15 text-violet-600 dark:text-violet-400 border-violet-500/30' },
    user: { label: 'User', className: 'bg-primary/15 text-primary border-primary/30' },
    assistant: { label: 'Assistant', className: 'bg-sky-500/15 text-sky-600 dark:text-sky-400 border-sky-500/30' },
  }[role] || { label: role, className: 'bg-muted text-muted-foreground' };

  return (
    <Badge variant="outline" className={cn('text-[10px] font-medium', config.className)}>
      {config.label}
    </Badge>
  );
}

export function TrajectoryPanel({ 
  iterations, 
  selectedIteration, 
  onSelectIteration 
}: TrajectoryPanelProps) {
  const [viewMode, setViewMode] = useState<'timeline' | 'messages'>('messages');

  const currentIteration = iterations[selectedIteration];

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border flex items-center justify-between bg-muted/30">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-sky-500/10 border border-sky-500/30 flex items-center justify-center">
            <span className="text-sky-500 text-sm">◈</span>
          </div>
          <div>
            <h2 className="font-semibold text-sm">Trajectory</h2>
            <p className="text-[11px] text-muted-foreground">
              Iteration {selectedIteration + 1} of {iterations.length}
            </p>
          </div>
        </div>
        <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as 'timeline' | 'messages')}>
          <TabsList className="h-7">
            <TabsTrigger value="messages" className="text-[11px] px-2.5 h-6">Messages</TabsTrigger>
            <TabsTrigger value="timeline" className="text-[11px] px-2.5 h-6">Timeline</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* Content */}
      <ScrollArea className="flex-1">
        {viewMode === 'timeline' ? (
          <div className="p-4">
            <div className="relative">
              {/* Timeline line */}
              <div className="absolute left-[18px] top-2 bottom-2 w-0.5 bg-border" />
              
              {/* Iterations */}
              <div className="space-y-3">
                {iterations.map((iter, idx) => {
                  const hasFinal = iter.final_answer !== null;
                  const hasCode = iter.code_blocks.length > 0;
                  const hasError = iter.code_blocks.some(b => b.result?.stderr);
                  const isSelected = selectedIteration === idx;
                  
                  return (
                    <div
                      key={idx}
                      onClick={() => onSelectIteration(idx)}
                      className={cn(
                        'relative pl-10 cursor-pointer group transition-all',
                        isSelected && 'scale-[1.01]'
                      )}
                    >
                      {/* Timeline dot */}
                      <div 
                        className={cn(
                          'absolute left-2.5 top-3 w-4 h-4 rounded-full border-2 flex items-center justify-center transition-all z-10',
                          isSelected
                            ? 'bg-primary border-primary scale-110 shadow-lg shadow-primary/30'
                            : hasFinal
                              ? 'bg-emerald-500 border-emerald-500'
                              : hasError
                                ? 'bg-destructive border-destructive'
                                : 'bg-card border-border group-hover:border-primary/50'
                        )}
                      >
                        <span className={cn(
                          'text-[9px] font-bold',
                          isSelected || hasFinal || hasError ? 'text-white' : 'text-muted-foreground'
                        )}>
                          {idx + 1}
                        </span>
                      </div>
                      
                      {/* Card */}
                      <Card className={cn(
                        'transition-all border',
                        isSelected
                          ? 'border-primary/50 bg-primary/5 shadow-md'
                          : 'hover:border-primary/30 hover:bg-muted/30'
                      )}>
                        <div className="p-3">
                          <div className="flex items-center gap-2 mb-2 flex-wrap">
                            <span className="text-[11px] font-medium text-foreground">
                              Iteration {iter.iteration}
                            </span>
                            {hasCode && (
                              <Badge variant="secondary" className="text-[9px] px-1.5 py-0 h-4">
                                {iter.code_blocks.length} code
                              </Badge>
                            )}
                            {hasFinal && (
                              <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 text-[9px] px-1.5 py-0 h-4">
                                ✓ Answer
                              </Badge>
                            )}
                            {hasError && (
                              <Badge variant="destructive" className="text-[9px] px-1.5 py-0 h-4">
                                Error
                              </Badge>
                            )}
                            <span className="text-[10px] text-muted-foreground ml-auto">
                              {new Date(iter.timestamp).toLocaleTimeString()}
                            </span>
                          </div>
                          
                          <p className="text-[12px] text-muted-foreground line-clamp-2 leading-relaxed">
                            {iter.response.slice(0, 120)}
                            {iter.response.length > 120 ? '...' : ''}
                          </p>
                        </div>
                      </Card>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        ) : (
          <div className="p-4 space-y-4">
            {/* Prompt messages */}
            {currentIteration?.prompt.map((msg, idx) => (
              <div 
                key={idx}
                className={cn(
                  'message-bubble',
                  msg.role === 'system' && 'message-bubble-system',
                  msg.role === 'user' && 'message-bubble-user',
                  msg.role === 'assistant' && 'message-bubble-assistant'
                )}
              >
                <div className="flex items-center gap-2 mb-3">
                  <RoleBadge role={msg.role} />
                  {msg.role === 'system' && (
                    <span className="text-[10px] text-muted-foreground">Initial prompt</span>
                  )}
                </div>
                <div className="prose-trajectory">
                  <pre className="whitespace-pre-wrap font-mono text-foreground/90 text-[13px] leading-relaxed">
                    {formatMessageContent(msg.content)}
                  </pre>
                </div>
              </div>
            ))}
            
            {/* Current response */}
            {currentIteration?.response && (
              <div className="message-bubble message-bubble-assistant">
                <div className="flex items-center gap-2 mb-3">
                  <RoleBadge role="assistant" />
                  <span className="text-[10px] text-muted-foreground">
                    Response • Iteration {currentIteration.iteration}
                  </span>
                  {currentIteration.code_blocks.length > 0 && (
                    <Badge variant="secondary" className="text-[9px] px-1.5 py-0 h-4 ml-auto">
                      {currentIteration.code_blocks.length} code block{currentIteration.code_blocks.length !== 1 ? 's' : ''}
                    </Badge>
                  )}
                </div>
                <div className="prose-trajectory">
                  <pre className="whitespace-pre-wrap font-mono text-foreground/90 text-[13px] leading-relaxed">
                    {formatMessageContent(currentIteration.response)}
                  </pre>
                </div>
              </div>
            )}

            {/* Final answer highlight */}
            {currentIteration?.final_answer && (
              <div className="rounded-xl border-2 border-emerald-500/50 bg-emerald-500/10 p-4">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-6 h-6 rounded-full bg-emerald-500 flex items-center justify-center">
                    <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <span className="font-semibold text-emerald-600 dark:text-emerald-400 text-sm">
                    Final Answer
                  </span>
                </div>
                <p className="text-[15px] font-medium text-foreground leading-relaxed">
                  {extractFinalAnswer(currentIteration.final_answer)}
                </p>
              </div>
            )}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
