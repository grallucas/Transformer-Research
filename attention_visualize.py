"""Notebook output helpers for inspecting generated attention."""

from IPython.display import HTML, Javascript
import html
import json
import uuid


def wrapped_output(text):
    escaped_text = html.escape(text)

    return HTML(
        f"""
        <pre style="
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            word-break: break-word;
            font-family: monospace;
            margin: 0;
        ">{escaped_text}</pre>
        """
    )


def attention_viewer(generated_text, prompt_length, attention_steps, n_layers, n_heads):
    viewer_id = "attention-viewer-" + uuid.uuid4().hex
    layer_options = "".join(
        f'<option value="{i}">Layer {i + 1}</option>' for i in range(n_layers)
    )
    head_options = '<option value="all">All heads</option>' + "".join(
        f'<option value="{i}">Head {i + 1}</option>' for i in range(n_heads)
    )

    characters = []
    for index, character in enumerate(generated_text):
        if character == "\n":
            label = "↵"
            line_break = "<br>"
        elif character == "\t":
            label = "⇥"
            line_break = ""
        elif character == " ":
            label = "&nbsp;"
            line_break = ""
        else:
            label = html.escape(character)
            line_break = ""

        generated = index >= prompt_length
        step_attribute = f' data-step="{index - prompt_length}"' if generated else ""
        token_class = "attn-token generated" if generated else "attn-token prompt"
        tabindex = ' tabindex="0"' if generated else ""
        characters.append(
            f'<span class="{token_class}" data-index="{index}"{step_attribute}'
            f"{tabindex}>"
            f'<span class="char">{label}</span><span class="dots" aria-hidden="true"></span>'
            f'</span>{line_break}'
        )

    styles = r"""
    #__ID__ { font-family: system-ui, sans-serif; color: inherit; }
    #__ID__ .controls { display: flex; flex-wrap: wrap; align-items: center; gap: .6em; margin-bottom: .7em; }
    #__ID__ select { font: inherit; padding: .2em .4em; }
    #__ID__ .status { margin: .4em 0; font-size: .9em; }
    #__ID__ .legend { display: flex; flex-wrap: wrap; gap: .8em; align-items: center; margin: .5em 0; font-size: .8em; }
    #__ID__ .legend-item { display: inline-flex; align-items: center; gap: .3em; }
    #__ID__ .text { max-height: 28em; overflow: auto; white-space: normal; line-height: 2.1; font: 14px/2.1 monospace; padding: .6em; border: 1px solid #8885; border-radius: .5em; }
    #__ID__ .attn-token { position: relative; display: inline-block; min-width: .7em; text-align: center; vertical-align: baseline; }
    #__ID__ .attn-token.generated { cursor: pointer; }
    #__ID__ .attn-token.selected .char { background: #f5c54299; border-radius: .2em; }
    #__ID__ .attn-token.generated:hover .char, #__ID__ .attn-token.generated:focus .char { outline: 1px solid #888; border-radius: .2em; }
    #__ID__ .dots { position: absolute; left: 50%; bottom: -.48em; transform: translateX(-50%); width: 9px; height: 9px; display: grid; grid-template-columns: repeat(2, 4px); grid-template-rows: repeat(2, 4px); gap: 1px; }
    #__ID__ .dot { display: block; width: 4px; height: 4px; background: currentColor; }
    #__ID__ .shape-0 { border-radius: 50%; }
    #__ID__ .shape-1 { border-radius: 1px; }
    #__ID__ .shape-2 { width: 3px; height: 3px; margin: .5px; transform: rotate(45deg); }
    #__ID__ .shape-3 { box-sizing: border-box; border: 1px solid currentColor; border-radius: 50%; background: transparent; }
    #__ID__ .hint { font-size: .8em; opacity: .8; margin-top: .5em; }
    """.replace("__ID__", viewer_id)

    markup = (
        f'<div id="{viewer_id}">'
        '<div class="controls">'
        f'<label>Layer <select class="layer-select">{layer_options}</select></label>'
        f'<label>Heads <select class="head-select">{head_options}</select></label>'
        '</div>'
        '<div class="status">Loading attention viewer…</div>'
        f'<div class="text">{"".join(characters)}</div>'
        '<div class="legend"></div>'
        '<div class="hint">Hover or focus a generated character to see attention used to predict it. '
        'Dots appear for probabilities of at least 2%; opacity is relative to the strongest '
        'position within each head. Hover a dot for its probability.</div>'
        '</div>'
    )

    steps = [
        [start, [
            [[round(float(probability), 5) for probability in head] for head in layer]
            for layer in probabilities
        ]]
        for start, probabilities in attention_steps
    ]
    script = r"""
    (function boot(attempt) {
        const root = document.getElementById(__VIEWER_ID__);
        if (!root) {
            if (attempt < 20) setTimeout(() => boot(attempt + 1), 50);
            return;
        }
        const steps = __ATTENTION_STEPS__;
        const nLayers = __N_LAYERS__;
        const nHeads = __N_HEADS__;
        const threshold = 0.02;
        const layerSelect = root.querySelector('.layer-select');
        const headSelect = root.querySelector('.head-select');
        const status = root.querySelector('.status');
        const legend = root.querySelector('.legend');
        const tokens = Array.from(root.querySelectorAll('.attn-token'));
        let selectedStep = -1;
        let marked = [];

        function color(layer, head) {
            const hue = (20 + 360 * layer / nLayers) % 360;
            const lightness = 45 + 7 * (head % 4);
            return `oklch(${lightness}% 0.16 ${hue})`;
        }
        function makeDot(layer, head, probability, maximum) {
            const dot = document.createElement('span');
            dot.className = `dot shape-${head % 4}`;
            dot.style.color = color(layer, head);
            dot.style.opacity = String(0.25 + 0.75 * Math.sqrt(probability / maximum));
            dot.style.gridColumn = String(head % 2 + 1);
            dot.style.gridRow = String(Math.floor(head % 4 / 2) + 1);
            dot.title = `Layer ${layer + 1}, head ${head + 1}: ${(100 * probability).toFixed(2)}%`;
            return dot;
        }
        function updateLegend(layer, heads) {
            legend.replaceChildren();
            for (const head of heads) {
                const item = document.createElement('span');
                item.className = 'legend-item';
                const dot = makeDot(layer, head, 1, 1);
                item.append(dot, document.createTextNode(`L${layer + 1} H${head + 1}`));
                legend.append(item);
            }
        }
        function render() {
            if (selectedStep < 0) {
                status.textContent = 'No generated characters to inspect.';
                return;
            }
            for (const index of marked) {
                tokens[index].querySelector('.dots').replaceChildren();
            }
            marked = [];
            const layer = Number(layerSelect.value);
            const heads = headSelect.value === 'all'
                ? Array.from({length: nHeads}, (_, i) => i)
                : [Number(headSelect.value)];
            updateLegend(layer, heads);
            const [start, probabilities] = steps[selectedStep];
            for (const head of heads) {
                const row = probabilities[layer][head];
                const maximum = Math.max(...row);
                for (let offset = 0; offset < row.length; offset++) {
                    const probability = row[offset];
                    if (probability < threshold) continue;
                    const index = start + offset;
                    tokens[index].querySelector('.dots').append(
                        makeDot(layer, head, probability, maximum)
                    );
                    marked.push(index);
                }
            }
            const selectedIndex = __PROMPT_LENGTH__ + selectedStep;
            const selectedToken = tokens[selectedIndex];
            status.textContent = `Attention used to predict character ${selectedStep + 1} of ${steps.length}`;
            root.querySelector('.attn-token.selected')?.classList.remove('selected');
            selectedToken.classList.add('selected');
        }
        function selectToken(event) {
            const token = event.target.closest('.attn-token[data-step]');
            if (!token || !root.contains(token)) return;
            const step = Number(token.dataset.step);
            if (step !== selectedStep) {
                selectedStep = step;
                render();
            }
        }
        root.addEventListener('mouseover', selectToken);
        root.addEventListener('focusin', selectToken);
        root.addEventListener('click', selectToken);
        layerSelect.addEventListener('change', render);
        headSelect.addEventListener('change', render);
        selectedStep = steps.length - 1;
        render();
    })(0);
    """
    script = (
        script.replace("__VIEWER_ID__", json.dumps(viewer_id))
        .replace("__ATTENTION_STEPS__", json.dumps(steps, separators=(",", ":")))
        .replace("__N_LAYERS__", str(n_layers))
        .replace("__N_HEADS__", str(n_heads))
        .replace("__PROMPT_LENGTH__", str(prompt_length))
    )
    return HTML(f"<style>{styles}</style>{markup}"), Javascript(script)
