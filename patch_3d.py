import sys

with open(r'src\drone_interceptor\dashboard\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Find the start of the function and the start of _build_map_deck
match = re.search(r'(def _build_3d_figure.*?)(def _build_map_deck)', content, flags=re.DOTALL)
if not match:
    print('Could not find function bounds')
    sys.exit(1)

old_func = match.group(1)

new_func = r'''def _build_3d_figure(simulation: dict[str, Any], upto_index: int | None = None) -> go.Figure:
    target = np.asarray(simulation["target_positions"], dtype=float)
    interceptor = np.asarray(simulation["interceptor_positions"], dtype=float)
    drifted = np.asarray(simulation["drifted_positions"], dtype=float)
    fused = np.asarray(simulation["fused_positions"], dtype=float)
    if upto_index is not None:
        limit = upto_index + 1
        target = target[:limit]
        interceptor = interceptor[:limit]
        drifted = drifted[:limit]
        fused = fused[:limit]
    comparison = simulation["comparison"]

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=target[:, 0], y=target[:, 1], mode="lines", name="Target True", line=dict(color="#ff845c", width=6)))
    figure.add_trace(go.Scatter(x=interceptor[:, 0], y=interceptor[:, 1], mode="lines", name="Interceptor", line=dict(color="#5ed2ff", width=6)))
    figure.add_trace(go.Scatter(x=fused[:, 0], y=fused[:, 1], mode="lines", name="EKF Filtered", line=dict(color="#73f0a0", width=5)))
    figure.add_trace(go.Scatter(x=drifted[:, 0], y=drifted[:, 1], mode="lines", name="Raw Drifted", line=dict(color="#ff4b4b", width=4, dash="dash")))

    if len(target) >= 2 and len(interceptor) >= 2:
        target_head = target[-1]
        interceptor_head = interceptor[-1]
        target_vec = target[-1] - target[-2]
        interceptor_vec = interceptor[-1] - interceptor[-2]
        figure.add_trace(
            go.Scatter(
                x=[target_head[0], target_head[0] + target_vec[0] * 4.0],
                y=[target_head[1], target_head[1] + target_vec[1] * 4.0],
                mode="lines+markers",
                name="Target Velocity",
                line=dict(color="#ff845c", width=8),
                marker=dict(size=3, color="#ff845c"),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[interceptor_head[0], interceptor_head[0] + interceptor_vec[0] * 4.0],
                y=[interceptor_head[1], interceptor_head[1] + interceptor_vec[1] * 4.0],
                mode="lines+markers",
                name="Interceptor Velocity",
                line=dict(color="#5ed2ff", width=8),
                marker=dict(size=3, color="#5ed2ff"),
            )
        )

    if comparison is not None:
        comparison_interceptor = np.asarray(comparison["interceptor_positions"], dtype=float)
        if upto_index is not None:
            comparison_interceptor = comparison_interceptor[: upto_index + 1]
        figure.add_trace(
            go.Scatter(
                x=comparison_interceptor[:, 0],
                y=comparison_interceptor[:, 1],
                mode="lines",
                name="Without Drift",
                line=dict(color="#7bf7a5", width=4, dash="dot"),
            )
        )

    if simulation["success"] and len(interceptor) > 0 and (upto_index is None or upto_index >= len(simulation["times"]) - 1):
        point = interceptor[-1]
        figure.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode="markers", name="Intercept", marker=dict(size=7, color="#ff5f99", symbol="diamond")))

    frames = []
    for index in range(len(target)):
        frame_traces = [
            go.Scatter(x=target[: index + 1, 0], y=target[: index + 1, 1], mode="lines", line=dict(color="#ff845c", width=6)),
            go.Scatter(x=interceptor[: index + 1, 0], y=interceptor[: index + 1, 1], mode="lines", line=dict(color="#5ed2ff", width=6)),
            go.Scatter(x=fused[: index + 1, 0], y=fused[: index + 1, 1], mode="lines", line=dict(color="#73f0a0", width=5)),
            go.Scatter(x=drifted[: index + 1, 0], y=drifted[: index + 1, 1], mode="lines", line=dict(color="#ff4b4b", width=4, dash="dash")),
        ]
        if comparison is not None:
            comparison_interceptor = np.asarray(comparison["interceptor_positions"], dtype=float)
            frame_traces.append(
                go.Scatter(
                    x=comparison_interceptor[: index + 1, 0],
                    y=comparison_interceptor[: index + 1, 1],
                    mode="lines",
                    line=dict(color="#7bf7a5", width=4, dash="dot"),
                )
            )
        frames.append(go.Frame(data=frame_traces, name=str(index)))
    figure.frames = frames

    figure.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(title="X [m]", backgroundcolor="rgba(0,0,0,0)", gridcolor="#2a3948"),
        yaxis=dict(title="Y [m]", backgroundcolor="rgba(0,0,0,0)", gridcolor="#2a3948", scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", y=1.02, x=0.0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=1.02,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 55, "redraw": True}, "fromcurrent": True}],
                    )
                ],
            )
        ],
    )
    return figure


'''

content = content.replace(old_func, new_func)

with open(r'src\drone_interceptor\dashboard\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Success')
