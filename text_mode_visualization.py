import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def analyze_reasoning_vs_answering(
    model,
    completion_text: str,
    reasoning_prompt: str = '<｜User｜>What is the seventh prime number?<｜Assistant｜><think>\n',
    answering_prompt: str = '<｜User｜>What is the seventh prime number?<｜Assistant｜><think>\n</think>\n\n',
    debug: bool = False,
    simple_debug: bool = False
) -> Dict:
    """
    Analyze how 'reasoning-like' vs 'answering-like' each token in a completion is.
    
    Args:
        model: HookedTransformer model
        completion_text: The text to analyze
        reasoning_prompt: Prompt that encourages reasoning behavior
        answering_prompt: Prompt that encourages direct answering
        debug: Print debugging information
    
    Returns:
        Dictionary with analysis results and visualization data
    """
    
    model.eval()
    
    if simple_debug or debug:
        print(f"\n{'='*60}")
        print("SIMPLE DEBUG: Analyzing reasoning vs answering")
        print(f"{'='*60}")
        print(f"Completion text: '{completion_text}'")
        print(f"Completion length: {len(completion_text)} chars")
    
    # Tokenize WITHOUT BOS tokens - this is crucial!
    completion_tokens = model.to_tokens(completion_text, prepend_bos=False)[0]  # Remove batch dim
    completion_str_tokens = model.to_str_tokens(completion_text, prepend_bos=False)
    
    if simple_debug or debug:
        print(f"Completion tokens: {len(completion_tokens)} tokens")
        print(f"First 5 tokens: {completion_str_tokens[:5]}")
        print(f"Last 5 tokens: {completion_str_tokens[-5:]}")
    
    # For each position, we'll compute log probabilities under both contexts
    reasoning_logprobs = []
    answering_logprobs = []
    reasoning_context = reasoning_prompt + completion_text
    answering_context = answering_prompt + completion_text
    
    if simple_debug or debug:
        print(f"\nReasoning context ({len(reasoning_context)} chars):")
        print(f"'{reasoning_context}'")
        print(f"\nAnswering context ({len(answering_context)} chars):")
        print(f"'{answering_context}'")
    
    # Get full context tokens (WITH BOS for the full contexts, since that's what the model expects)
    reasoning_tokens = model.to_tokens(reasoning_context, prepend_bos=True)[0]  # Model needs BOS for inference
    answering_tokens = model.to_tokens(answering_context, prepend_bos=True)[0]
    
    # Get prompt lengths (WITHOUT BOS for accurate positioning)
    reasoning_prompt_tokens = model.to_tokens(reasoning_prompt, prepend_bos=False)[0]
    answering_prompt_tokens = model.to_tokens(answering_prompt, prepend_bos=False)[0]
    reasoning_prompt_len = len(reasoning_prompt_tokens)
    answering_prompt_len = len(answering_prompt_tokens)
    
    if simple_debug or debug:
        print(f"\nTokenization summary:")
        print(f"  Reasoning prompt: {reasoning_prompt_len} tokens (no BOS)")
        print(f"  Answering prompt: {answering_prompt_len} tokens (no BOS)")
        print(f"  Full reasoning context: {len(reasoning_tokens)} tokens (with BOS)")
        print(f"  Full answering context: {len(answering_tokens)} tokens (with BOS)")
        print(f"  Expected completion start positions:")
        print(f"    Reasoning: {1 + reasoning_prompt_len} (1 for BOS + prompt length)")
        print(f"    Answering: {1 + answering_prompt_len} (1 for BOS + prompt length)")
    
    # Verify that the completion tokens match what we expect in the full contexts
    if simple_debug or debug:
        reasoning_completion_start = 1 + reasoning_prompt_len
        answering_completion_start = 1 + answering_prompt_len
        
        # Check if completion tokens match in reasoning context
        reasoning_completion_tokens = reasoning_tokens[reasoning_completion_start:reasoning_completion_start + len(completion_tokens)]
        answering_completion_tokens = answering_tokens[answering_completion_start:answering_completion_start + len(completion_tokens)]
        
        reasoning_match = torch.equal(completion_tokens, reasoning_completion_tokens)
        answering_match = torch.equal(completion_tokens, answering_completion_tokens)
        
        print(f"\nToken matching verification:")
        print(f"  Reasoning context completion tokens match: {reasoning_match}")
        print(f"  Answering context completion tokens match: {answering_match}")
        
        if not reasoning_match or not answering_match:
            print(f"  WARNING: Token mismatch detected!")
            print(f"  Original completion tokens: {completion_tokens[:5]}...")
            print(f"  Reasoning context tokens: {reasoning_completion_tokens[:5]}...")
            print(f"  Answering context tokens: {answering_completion_tokens[:5]}...")
    
    if debug:
        print(f"Reasoning prompt length: {reasoning_prompt_len} tokens (no BOS)")
        print(f"Answering prompt length: {answering_prompt_len} tokens (no BOS)")
        print(f"Full reasoning context length: {len(reasoning_tokens)} tokens (with BOS)")
        print(f"Full answering context length: {len(answering_tokens)} tokens (with BOS)")
    
    # Get logits for both contexts
    with torch.no_grad():
        reasoning_logits = model(reasoning_tokens.unsqueeze(0))  # Add batch dim
        answering_logits = model(answering_tokens.unsqueeze(0))
    
    if simple_debug or debug:
        print(f"\nLogits shapes:")
        print(f"  Reasoning logits: {reasoning_logits.shape}")
        print(f"  Answering logits: {answering_logits.shape}")
    
    # Extract logprobs for each completion token
    for i, token_id in enumerate(completion_tokens):
        # Position in the full context (after BOS + prompt)
        # We add +1 to account for BOS token in the full context
        reasoning_pos = 1 + reasoning_prompt_len + i - 1  # -1 because we predict next token
        answering_pos = 1 + answering_prompt_len + i - 1
        
        if reasoning_pos >= 0 and reasoning_pos < reasoning_logits.shape[1]:
            reasoning_logprob = torch.log_softmax(reasoning_logits[0, reasoning_pos], dim=-1)[token_id].item()
        else:
            reasoning_logprob = float('-inf')
            if simple_debug or debug:
                print(f"WARNING: reasoning_pos {reasoning_pos} out of bounds for token {i}")
            
        if answering_pos >= 0 and answering_pos < answering_logits.shape[1]:
            answering_logprob = torch.log_softmax(answering_logits[0, answering_pos], dim=-1)[token_id].item()
        else:
            answering_logprob = float('-inf')
            if simple_debug or debug:
                print(f"WARNING: answering_pos {answering_pos} out of bounds for token {i}")
        
        reasoning_logprobs.append(reasoning_logprob)
        answering_logprobs.append(answering_logprob)
        
        if debug and i < 5:  # Print first few
            token_str = completion_str_tokens[i]  # Use pre-computed string tokens
            print(f"Token {i}: '{token_str}' | Reasoning: {reasoning_logprob:.3f} | Answering: {answering_logprob:.3f}")
            print(f"  Reasoning pos: {reasoning_pos}, Answering pos: {answering_pos}")
    
    if simple_debug or debug:
        print(f"\nCompleted analysis of {len(completion_tokens)} tokens")
        print(f"{'='*60}")
    
    # Convert to numpy arrays
    reasoning_logprobs = np.array(reasoning_logprobs)
    answering_logprobs = np.array(answering_logprobs)
    
    # Compute ratios and metrics
    # Ratio > 1 means more "reasoning-like", < 1 means more "answering-like"
    logprob_diff = reasoning_logprobs - answering_logprobs  # log(P_reasoning/P_answering)
    prob_ratio = np.exp(logprob_diff)  # P_reasoning/P_answering
    
    # Normalized score: -1 (very answering-like) to +1 (very reasoning-like)
    reasoning_score = np.tanh(logprob_diff)
    
    return {
        'tokens': completion_str_tokens,
        'token_ids': completion_tokens.tolist(),
        'reasoning_logprobs': reasoning_logprobs,
        'answering_logprobs': answering_logprobs,
        'logprob_diff': logprob_diff,
        'prob_ratio': prob_ratio,
        'reasoning_score': reasoning_score,
        'completion_text': completion_text,
        'reasoning_prompt': reasoning_prompt,
        'answering_prompt': answering_prompt
    }

def visualize_reasoning_analysis(analysis_result: Dict, 
                               title: str = "Reasoning vs Answering Analysis",
                               show_plotly: bool = False,  # Default to False to avoid dependency issues
                               show_matplotlib: bool = True) -> None:
    """
    Create visualizations of the reasoning vs answering analysis.
    """
    
    tokens = analysis_result['tokens']
    reasoning_score = analysis_result['reasoning_score']
    prob_ratio = analysis_result['prob_ratio']
    logprob_diff = analysis_result['logprob_diff']
    
    if show_plotly:
        try:
            # Interactive Plotly visualization
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=['Token-by-Token Reasoning Score', 'Probability Ratio', 'Text with Background Colors'],
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}]],
                vertical_spacing=0.1
            )
            
            # Score plot
            colors = ['red' if score < -0.2 else 'blue' if score > 0.2 else 'gray' for score in reasoning_score]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(tokens))),
                    y=reasoning_score,
                    mode='markers+lines',
                    marker=dict(color=colors, size=8),
                    text=[f"'{token}'" for token in tokens],
                    hovertemplate='Token: %{text}<br>Reasoning Score: %{y:.3f}<extra></extra>',
                    name='Reasoning Score'
                ),
                row=1, col=1
            )
            
            # Add horizontal lines for reference
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)
            fig.add_hline(y=0.5, line_dash="dot", line_color="blue", opacity=0.3, row=1, col=1)
            fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.3, row=1, col=1)
            
            # Probability ratio plot (log scale)
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(tokens))),
                    y=np.log10(np.maximum(prob_ratio, 1e-10)),  # Avoid log(0)
                    mode='markers+lines',
                    marker=dict(color=colors, size=8),
                    text=[f"'{token}'" for token in tokens],
                    hovertemplate='Token: %{text}<br>Log10(Ratio): %{y:.3f}<extra></extra>',
                    name='Log10(Prob Ratio)'
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
            
            # Text visualization with colored backgrounds
            # Create a heatmap-style visualization
            reasoning_score_normalized = (reasoning_score + 1) / 2  # Scale to 0-1
            
            fig.add_trace(
                go.Heatmap(
                    z=[reasoning_score_normalized],
                    x=list(range(len(tokens))),
                    y=[0],
                    colorscale=[[0, 'red'], [0.5, 'white'], [1, 'blue']],
                    showscale=True,
                    colorbar=dict(title="Reasoning Score", y=0.15, len=0.3),
                    hovertemplate='Token %{x}: %{text}<br>Score: %{customdata:.3f}<extra></extra>',
                    text=[tokens],
                    customdata=reasoning_score
                ),
                row=3, col=1
            )
            
            fig.update_layout(
                title=title,
                height=800,
                showlegend=False
            )
            
            fig.update_xaxes(title_text="Token Position", row=3, col=1)
            fig.update_yaxes(title_text="Reasoning Score (-1=Answering, +1=Reasoning)", row=1, col=1)
            fig.update_yaxes(title_text="Log10(P_reasoning/P_answering)", row=2, col=1)
            fig.update_yaxes(title_text="", showticklabels=False, row=3, col=1)
            
            # Try to show, but handle potential errors
            try:
                fig.show()
            except Exception as e:
                print(f"Could not display Plotly figure: {e}")
                print("Consider installing: pip install nbformat>=4.2.0")
                print("Falling back to matplotlib visualization...")
                show_matplotlib = True
        except Exception as e:
            print(f"Plotly visualization failed: {e}")
            print("Falling back to matplotlib visualization...")
            show_matplotlib = True
    
    if show_matplotlib:
        # Static matplotlib visualization
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Token score plot
        colors = ['red' if score < -0.2 else 'blue' if score > 0.2 else 'gray' for score in reasoning_score]
        axes[0].scatter(range(len(tokens)), reasoning_score, c=colors, alpha=0.7, s=50)
        axes[0].plot(range(len(tokens)), reasoning_score, alpha=0.5, color='black')
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0].axhline(y=0.5, color='blue', linestyle=':', alpha=0.3)
        axes[0].axhline(y=-0.5, color='red', linestyle=':', alpha=0.3)
        axes[0].set_ylabel('Reasoning Score')
        axes[0].set_title('Token-by-Token Reasoning vs Answering Score')
        axes[0].grid(True, alpha=0.3)
        
        # Probability ratio plot
        axes[1].scatter(range(len(tokens)), np.log10(np.maximum(prob_ratio, 1e-10)), c=colors, alpha=0.7, s=50)
        axes[1].plot(range(len(tokens)), np.log10(np.maximum(prob_ratio, 1e-10)), alpha=0.5, color='black')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Log10(P_reasoning/P_answering)')
        axes[1].set_title('Probability Ratio (Log Scale)')
        axes[1].grid(True, alpha=0.3)
        
        # Text with background colors
        text_viz = axes[2]
        text_viz.set_xlim(-0.5, len(tokens) - 0.5)
        text_viz.set_ylim(-0.5, 0.5)
        
        # Create background colors for each token
        for i, (token, score) in enumerate(zip(tokens, reasoning_score)):
            # Normalize score to color intensity
            if score > 0:
                color = plt.cm.Blues(min(abs(score), 1.0))
            else:
                color = plt.cm.Reds(min(abs(score), 1.0))
            
            # Add colored background rectangle
            rect = plt.Rectangle((i-0.4, -0.4), 0.8, 0.8, facecolor=color, alpha=0.7)
            text_viz.add_patch(rect)
            
            # Add token text
            text_viz.text(i, 0, token, ha='center', va='center', fontsize=8, 
                         fontweight='bold', color='white' if abs(score) > 0.5 else 'black')
        
        text_viz.set_xlabel('Token Position')
        text_viz.set_title('Text Visualization (Blue=Reasoning-like, Red=Answering-like)')
        text_viz.set_yticks([])
        text_viz.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Always show a simple text-based visualization
    print_text_visualization(analysis_result)

def print_text_visualization(analysis_result: Dict) -> None:
    """
    Print a simple text-based visualization that works in any environment.
    """
    tokens = analysis_result['tokens']
    reasoning_score = analysis_result['reasoning_score']
    
    print(f"\n{'='*80}")
    print("TEXT-BASED VISUALIZATION")
    print(f"{'='*80}")
    print("Legend: [R] = Reasoning-like, [A] = Answering-like, [ ] = Neutral")
    print(f"{'='*80}")
    
    # Print tokens with indicators
    line_length = 0
    for i, (token, score) in enumerate(zip(tokens, reasoning_score)):
        # Determine indicator
        if score > 0.2:
            indicator = "R"
            color_code = "\033[94m"  # Blue
        elif score < -0.2:
            indicator = "A"
            color_code = "\033[91m"  # Red
        else:
            indicator = " "
            color_code = "\033[0m"   # Reset
        
        # Format token with indicator and score
        token_display = f"{color_code}[{indicator}]{token}\033[0m"
        score_display = f"({score:.2f})"
        
        # Print with line wrapping
        token_length = len(token) + len(score_display) + 4  # 4 for brackets and spaces
        if line_length + token_length > 80:
            print()  # New line
            line_length = 0
        
        print(token_display + score_display, end=" ")
        line_length += token_length + 1
    
    print("\n")
    
    # Print summary statistics
    reasoning_tokens = sum(1 for s in reasoning_score if s > 0.2)
    answering_tokens = sum(1 for s in reasoning_score if s < -0.2)
    neutral_tokens = len(reasoning_score) - reasoning_tokens - answering_tokens
    avg_score = np.mean(reasoning_score)
    
    print(f"Summary Statistics:")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Reasoning-like tokens [R]: {reasoning_tokens} ({reasoning_tokens/len(tokens)*100:.1f}%)")
    print(f"  Answering-like tokens [A]: {answering_tokens} ({answering_tokens/len(tokens)*100:.1f}%)")
    print(f"  Neutral tokens [ ]: {neutral_tokens} ({neutral_tokens/len(tokens)*100:.1f}%)")
    print(f"  Average reasoning score: {avg_score:.3f}")
    
    # Show most reasoning-like and answering-like tokens
    token_scores = list(zip(tokens, reasoning_score))
    token_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMost reasoning-like tokens:")
    for token, score in token_scores[:5]:
        if score > 0:
            print(f"  '{token}': {score:.3f}")
    
    print(f"\nMost answering-like tokens:")
    for token, score in reversed(token_scores[-5:]):
        if score < 0:
            print(f"  '{token}': {score:.3f}")
    
    print(f"{'='*80}")

def create_png_visualization(analysis_result: Dict, 
                           output_file: str = "reasoning_analysis.png",
                           chars_per_line: int = 100,
                           font_size: int = 12) -> str:
    """
    Create a PNG image visualization with monospace font and colored backgrounds.
    Red = answering-like, Green = reasoning-like, White = neutral.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib import font_manager
    
    tokens = analysis_result['tokens']
    reasoning_score = analysis_result['reasoning_score']
    
    # Reconstruct the text for proper line wrapping
    full_text = ''.join(tokens)
    
    # Set up the figure with monospace font
    plt.rcParams['font.family'] = 'monospace'
    
    # Calculate figure dimensions
    lines = []
    current_line = ""
    current_scores = []
    line_scores = []
    
    token_idx = 0
    for token in tokens:
        if len(current_line) + len(token) <= chars_per_line:
            current_line += token
            current_scores.append(reasoning_score[token_idx])
        else:
            if current_line:  # Don't add empty lines
                lines.append(current_line)
                line_scores.append(current_scores)
            current_line = token
            current_scores = [reasoning_score[token_idx]]
        token_idx += 1
    
    # Add the last line
    if current_line:
        lines.append(current_line)
        line_scores.append(current_scores)
    
    # Create figure
    fig_width = max(12, chars_per_line * 0.1)
    fig_height = max(6, len(lines) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Set up the plot
    ax.set_xlim(0, chars_per_line)
    ax.set_ylim(0, len(lines))
    ax.set_aspect('equal')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Color mapping: red (-1) -> white (0) -> green (+1)
    def score_to_color(score):
        if score > 0:
            # Reasoning-like: interpolate from white to green
            intensity = min(abs(score), 1.0)
            return (1 - intensity * 0.7, 1.0, 1 - intensity * 0.7)  # White to green
        elif score < 0:
            # Answering-like: interpolate from white to red  
            intensity = min(abs(score), 1.0)
            return (1.0, 1 - intensity * 0.7, 1 - intensity * 0.7)  # White to red
        else:
            return (1.0, 1.0, 1.0)  # White for neutral
    
    # Draw each line with character-by-character coloring
    for line_idx, (line_text, scores) in enumerate(zip(lines, line_scores)):
        y_pos = len(lines) - line_idx - 1  # Flip y-axis so first line is at top
        
        # Track character position within the line
        char_pos = 0
        token_idx = 0
        
        # We need to map characters back to tokens for coloring
        remaining_line = line_text
        
        for token, score in zip(tokens, reasoning_score):
            if not remaining_line:
                break
                
            if remaining_line.startswith(token):
                # This token is at the start of remaining_line
                token_len = len(token)
                
                # Draw background rectangle for this token
                rect = patches.Rectangle(
                    (char_pos, y_pos), token_len, 1,
                    facecolor=score_to_color(score),
                    edgecolor='none',
                    alpha=0.8
                )
                ax.add_patch(rect)
                
                # Draw the token text
                ax.text(char_pos + token_len/2, y_pos + 0.5, token, 
                       ha='center', va='center', 
                       fontsize=font_size, 
                       fontfamily='monospace',
                       color='black',
                       weight='bold' if abs(score) > 0.5 else 'normal')
                
                char_pos += token_len
                remaining_line = remaining_line[token_len:]
            else:
                # Token doesn't match - this can happen with tokenization quirks
                # Just skip this token for this line
                continue
    
    # Add title and legend
    plt.suptitle(f'Reasoning vs Answering Analysis\n'
                f'Red = Answering-like, Green = Reasoning-like, White = Neutral', 
                fontsize=14, y=0.98)
    
    # Add color legend
    legend_y = -0.1
    legend_elements = [
        patches.Rectangle((0, 0), 1, 1, facecolor=(1.0, 0.3, 0.3), label='Answering-like'),
        patches.Rectangle((0, 0), 1, 1, facecolor=(1.0, 1.0, 1.0), edgecolor='black', label='Neutral'),
        patches.Rectangle((0, 0), 1, 1, facecolor=(0.3, 1.0, 0.3), label='Reasoning-like')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    # Add statistics text
    reasoning_tokens = sum(1 for s in reasoning_score if s > 0.2)
    answering_tokens = sum(1 for s in reasoning_score if s < -0.2)
    neutral_tokens = len(reasoning_score) - reasoning_tokens - answering_tokens
    avg_score = np.mean(reasoning_score)
    
    stats_text = (f"Stats: {len(tokens)} tokens | "
                 f"Reasoning: {reasoning_tokens} ({reasoning_tokens/len(tokens)*100:.1f}%) | "
                 f"Answering: {answering_tokens} ({answering_tokens/len(tokens)*100:.1f}%) | "
                 f"Neutral: {neutral_tokens} ({neutral_tokens/len(tokens)*100:.1f}%) | "
                 f"Avg: {avg_score:.3f}")
    
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10, style='italic')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_file

def create_simple_png_visualization(analysis_result: Dict, 
                                  output_file: str = "reasoning_analysis_simple.png") -> str:
    """
    Create a simpler PNG visualization that's more reliable.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    tokens = analysis_result['tokens']
    reasoning_score = analysis_result['reasoning_score']
    
    # Create a simple grid layout
    tokens_per_line = 20
    num_lines = (len(tokens) + tokens_per_line - 1) // tokens_per_line
    
    fig, ax = plt.subplots(figsize=(16, max(6, num_lines * 0.8)))
    
    # Color mapping function
    def score_to_color(score):
        if score > 0.2:
            return 'lightgreen'
        elif score < -0.2:
            return 'lightcoral'
        else:
            return 'lightgray'
    
    # Plot tokens in a grid
    for i, (token, score) in enumerate(zip(tokens, reasoning_score)):
        row = i // tokens_per_line
        col = i % tokens_per_line
        
        # Draw background rectangle
        rect = patches.Rectangle((col, num_lines - row - 1), 1, 1, 
                               facecolor=score_to_color(score), 
                               edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        
        # Add token text (truncate if too long)
        display_token = token[:6] + '...' if len(token) > 6 else token
        ax.text(col + 0.5, num_lines - row - 0.5, display_token,
               ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_xlim(0, tokens_per_line)
    ax.set_ylim(0, num_lines)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add title and legend
    plt.title('Reasoning vs Answering Token Analysis', fontsize=14, pad=20)
    
    # Create legend
    legend_elements = [
        patches.Rectangle((0, 0), 1, 1, facecolor='lightcoral', label='Answering-like'),
        patches.Rectangle((0, 0), 1, 1, facecolor='lightgray', label='Neutral'),
        patches.Rectangle((0, 0), 1, 1, facecolor='lightgreen', label='Reasoning-like')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    # Add statistics
    reasoning_tokens = sum(1 for s in reasoning_score if s > 0.2)
    answering_tokens = sum(1 for s in reasoning_score if s < -0.2)
    avg_score = np.mean(reasoning_score)
    
    stats_text = f"Total: {len(tokens)} | Reasoning: {reasoning_tokens} | Answering: {answering_tokens} | Avg: {avg_score:.3f}"
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_file

def create_html_visualization(analysis_result: Dict, output_file: str = "reasoning_analysis.html") -> str:
    """
    Create an HTML file with colored text visualization.
    """
    tokens = analysis_result['tokens']
    reasoning_score = analysis_result['reasoning_score']
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reasoning vs Answering Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .token {{ display: inline-block; padding: 2px 4px; margin: 1px; border-radius: 3px; }}
            .legend {{ margin: 20px 0; }}
            .legend-item {{ display: inline-block; margin-right: 20px; }}
            .stats {{ background: #f5f5f5; padding: 10px; margin: 20px 0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Reasoning vs Answering Analysis</h1>
        
        <div class="legend">
            <div class="legend-item"><span style="background: #ff6b6b; color: white; padding: 2px 8px;">■</span> Answering-like</div>
            <div class="legend-item"><span style="background: #ffffff; color: black; padding: 2px 8px; border: 1px solid #ccc;">■</span> Neutral</div>
            <div class="legend-item"><span style="background: #4dabf7; color: white; padding: 2px 8px;">■</span> Reasoning-like</div>
        </div>
        
        <div class="text-content">
    """
    
    for token, score in zip(tokens, reasoning_score):
        # Determine background color based on score
        if score > 0.2:
            intensity = min(abs(score), 1.0)
            bg_color = f"rgba(77, 171, 247, {intensity})"  # Blue for reasoning
            text_color = "white" if intensity > 0.5 else "black"
        elif score < -0.2:
            intensity = min(abs(score), 1.0)
            bg_color = f"rgba(255, 107, 107, {intensity})"  # Red for answering
            text_color = "white" if intensity > 0.5 else "black"
        else:
            bg_color = "#f8f9fa"  # Light gray for neutral
            text_color = "black"
        
        html_content += f'<span class="token" style="background-color: {bg_color}; color: {text_color};" title="Score: {score:.3f}">{token}</span>'
    
    html_content += f"""
        </div>
        
        <div class="stats">
            <h3>Statistics</h3>
            <p><strong>Total tokens:</strong> {len(tokens)}</p>
            <p><strong>Reasoning-like tokens (score > 0.2):</strong> {sum(1 for s in reasoning_score if s > 0.2)}</p>
            <p><strong>Answering-like tokens (score < -0.2):</strong> {sum(1 for s in reasoning_score if s < -0.2)}</p>
            <p><strong>Neutral tokens:</strong> {sum(1 for s in reasoning_score if -0.2 <= s <= 0.2)}</p>
            <p><strong>Average reasoning score:</strong> {np.mean(reasoning_score):.3f}</p>
        </div>
        
        <div class="prompts">
            <h3>Prompts Used</h3>
            <p><strong>Reasoning prompt:</strong> "{analysis_result['reasoning_prompt']}"</p>
            <p><strong>Answering prompt:</strong> "{analysis_result['answering_prompt']}"</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file

# Example usage function
def analyze_text_example(model, text: str, debug: bool = False, simple_debug: bool = False):
    """
    Example function showing how to use the analysis tools.
    """
    # Analyze the text
    analysis = analyze_reasoning_vs_answering(
        model=model,
        completion_text=text,
        reasoning_prompt='<｜User｜>What is the seventh prime number?<｜Assistant｜><think>\n',
        answering_prompt='<｜User｜>What is the seventh prime number?<｜Assistant｜><think>\n</think>\n\n',
        debug=debug,
        simple_debug=simple_debug
    )
    
    # Create PNG visualization
    png_file = create_simple_png_visualization(analysis)
    print(f"PNG visualization saved to: {png_file}")
    
    # Show text-based visualization
    print_text_visualization(analysis)
    
    return analysis

def analyze_text_simple(model, text: str, reasoning_prompt: str = None, answering_prompt: str = None, simple_debug: bool = False):
    """
    Simplified analysis function that creates PNG and shows text-based visualization.
    """
    if reasoning_prompt is None:
        reasoning_prompt = '<｜User｜>What is the seventh prime number?<｜Assistant｜><think>\n'
    if answering_prompt is None:
        answering_prompt = '<｜User｜>What is the seventh prime number?<｜Assistant｜><think>\n</think>\n\n'
    
    # Analyze the text
    analysis = analyze_reasoning_vs_answering(
        model=model,
        completion_text=text,
        reasoning_prompt=reasoning_prompt,
        answering_prompt=answering_prompt,
        debug=False,
        simple_debug=simple_debug
    )
    
    # Create PNG visualization
    png_file = create_simple_png_visualization(analysis)
    print(f"PNG visualization saved to: {png_file}")
    
    # Show text-based visualization
    print_text_visualization(analysis)
    
    return analysis