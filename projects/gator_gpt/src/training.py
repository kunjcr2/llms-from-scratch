import torch
from tokenization import *

###################################### LOSS FOR ONE BATCH
# Caluclates loss for a batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # For other devices
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

    # For A100s
    # with autocast(device_type="cuda", dtype=torch.bfloat16):
    #     logits = model(input_batch)
    #     loss = torch.nn.functional.cross_entropy(
    #         logits.flatten(0, 1),
    #         target_batch.flatten()
    #     )

    return loss
############################################################################

###################################### LOSS FOR ENTIRE LOADER
# Caluculates loss for ENTIRE data_loader which calls calc_loss_batch function inside itself
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
############################################################################

###################################### USE THIS TO EVALUATE MODEL DIRECLTY
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
  # basically returns the losses for training and validation
  model.eval()
  with torch.no_grad():
    return calc_loss_loader(train_loader, model, device, eval_iter), calc_loss_loader(val_loader, model, device, eval_iter)
############################################################################

####################################### GENERATING NEW TOKENS
def generate_text_simple(model, idx, max_new_tokens, context_size):

  # idx is (batch, n_tokens) array of indices in current context
  for _ in range(max_new_tokens):
    # If LLM suports only 5 tokens, and the context size is 10, then we only use last 5 toens as context.
    idx_cond = idx[:, -context_size:]

    # Gettings the predictions
    with torch.no_grad():
      # Reshape idx_cond to (batch_size, sequence_length, emb_dim)
      # idx_cond = idx_cond.unsqueeze(-1).repeat(1 , 1, model.norm1.scale.shape[0]) # Or model.att.d_in to get the embedding dimension
      logits = model(idx_cond) # (batch, num_tokens, vocab_size)

    # We take the last row. We dont do anything to the batches neither to the last dimension of the vocabularies, but take the last row
    logits = logits[:, -1, :] # (batch, vocab_size)

    # getting probablities from the logits. We can say something like 50% chances of this, 2% chances of this...
    probs = torch.softmax(logits, dim=-1) # (batch, vocab_size)

    # We see the highest value's index
    idx_next = torch.argmax(probs, dim=-1, keepdim=True) # (batch, 1)

    # Append the predicted token_id generated to the original index
    idx = torch.cat((idx, idx_next), dim=1) # (batch, num_tokens+1)

  return idx
############################################################################

###################################### GENERATING AND PRINTING SAMPLES
def generate_and_print_sample(model, tokenizer, device, start_context, context_size):
  # we print out, what the model is generating right now at the end of each epoch. Also, we print 50 items!
  model.eval()
  encoded = text_to_token_ids(start_context, tokenizer).to(device)

  with torch.no_grad():
    token_ids = generate_text_simple(
        model, encoded, 50, context_size
    )

  decoded = token_ids_to_text(token_ids, tokenizer)
  print(decoded.replace("\n", " "))
  model.train()
############################################################################
