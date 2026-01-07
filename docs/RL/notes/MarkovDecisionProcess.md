# Markov Decision Process

- **Episodic tasks** are those where a game or a goal can be achieved or not. Like Tic Tac Toe - we can loose or win. If the game ends, episode ends. **BUT** there are certain tasks where episodes doesnt make sense like Martian rover on Mars, they are called **Continuos Tasks**.

### **Finite Markov Decision Processes**:

So this is the property where we Assume or sometimes even prove that to get the action of next state, we ONLY need the previous state and not ALL the previous state.

Conffusin, right ? In chess, we dont need to know what happened in past, our current state is enough to do actions on the state while being in a middle of a conversation with someone, we need to know past conversation.

And the processes that holds this property of not needing to knowall the past states, are called _Markov Decicion Property_ !
