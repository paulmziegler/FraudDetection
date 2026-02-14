# DGA Layer (`src/models/dga_layers.py`)

## Core Functionality
The `dga_layers.py` module defines the custom Graph Neural Network layer: `DGAGNNLayer`. This is the mathematical heart of the project.

Standard GNNs (like GCN or GraphSAGE) treat all neighbors equally (or weighted by static edge weights). **DGA** treats neighbors differently based on their **Dynamic Group Label**.

If a neighbor is labeled "Group 0" (e.g., predicted benign in the last epoch), it is processed with weight matrix $W_0$. If it is "Group 1" (predicted fraud), it is processed with $W_1$. This allows the model to learn different aggregation patterns for "interaction with a fraudster" vs "interaction with a normal user."

## ELI5 Description
This layer is like a very prejudiced gossip listener. When it hears news (messages) from neighbors, it doesn't listen to everyone the same way.

It checks the label on the neighbor's forehead.
-   "Oh, you're labeled **Suspicious**? I'll process your story through my 'Suspicious Filter'."
-   "You're labeled **Safe**? I'll process your story through my 'Safe Filter'."

Then, it combines all these filtered stories together to update what it knows about the central person.

## Classes

### `DGAGNNLayer`
-   **Inherits**: `torch.nn.Module`
-   **Init**: Creates `self.neighbor_transforms`, a list of Linear layers, one for each group.
-   **Methods**:
    -   `forward(g, h, group_labels)`: Orchestrates the message passing. Stores `h` and `group` in the graph and calls `update_all`.
    -   `message_func(edges)`: Sends the source node's features (`h`) AND its group label (`group`) to the destination.
    -   `reduce_func(nodes)`:
        -   Receives a mailbox of messages.
        -   Sorts messages by their group.
        -   Applies `transform_0` to Group 0 messages, `transform_1` to Group 1 messages, etc.
        -   Sums the results (`scatter_add`) to get the final aggregated representation (`h_agg`).
