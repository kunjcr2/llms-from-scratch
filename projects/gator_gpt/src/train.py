from training import *

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, Patience):
    # Initialize tracking lists
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float('inf')
    PATIENCE = Patience
    patience_counter = 0

    # Check dataset size
    print(f"📊 Dataset info:")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Total steps per epoch: {len(train_loader)}")
    print(f"   Evaluation every {eval_freq} steps")
    print("="*50)

    for epoch in range(num_epochs):
        print(f"🔁 Starting epoch {epoch+1}/{num_epochs}")
        print("-"*30)
        model.train()

        epoch_steps = 0
        epoch_loss = 0.0

        for input_batch, target_batch in train_loader:

            if patience_counter >= PATIENCE:
              break

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1
            epoch_steps += 1
            epoch_loss += loss.item()

            # Print progress within epoch
            if epoch_steps % max(1, len(train_loader) // 5) == 0:  # Print 5 times per epoch
                avg_loss = epoch_loss / epoch_steps
                progress = epoch_steps / len(train_loader) * 100
                print(f"   Step {epoch_steps}/{len(train_loader)} ({progress:.1f}%) - "
                      f"Current loss: {loss.item():.3f}, Avg loss: {avg_loss:.3f}, patience level: {patience_counter}/5")

            # Evaluation during training
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                improvement = "✅" if val_loss < best_val_loss else "❌"
                print(f"   Eval at step {global_step}: Train {train_loss:.3f}, "
                      f"Val {val_loss:.3f} {improvement}, patience level: {patience_counter}/5")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                else:
                    patience_counter += 1

        # End of epoch summary
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else float('inf')
        print(f"✅ Epoch {epoch+1} complete: {epoch_steps} steps, avg loss: {avg_epoch_loss:.3f}")

        # Generate sample after each epoch
        print("🎯 Generated sample:")
        generate_and_print_sample(model, tokenizer, device, start_context, 1024)
        print("="*50)

    print(f"🏁 Training complete! Best validation loss: {best_val_loss:.3f}")
    return train_losses, val_losses, track_tokens_seen
