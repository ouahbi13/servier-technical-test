import argparse
from servier.model_1 import train_model_1, evaluate_model_1, predict_model_1
from servier.model_2 import train_model_2, evaluate_model_2, predict_model_2

def train(args):
    model_type = args.model_type
    epochs = args.epochs
    batch_size = args.batch_size
    
    if model_type == 'model_1':
        train_model_1(epochs=epochs, batch_size=batch_size)
    elif model_type == 'model_2':
        train_model_2(epochs=epochs, batch_size=batch_size)

    print(f"Training model with arguments: {args}")


def evaluate(args):
    model_type = args.model_type
    epochs = args.epochs
    batch_size = args.batch_size
    n_splits = args.n_splits
    
    if model_type == 'model_1':
        results = evaluate_model_1(epochs=epochs, batch_size=batch_size, n_splits=n_splits)
    elif model_type == 'model_2':
        results = evaluate_model_2(epochs=epochs, batch_size=batch_size, n_splits=n_splits)

    print('Cross-Val Results :', results)
    
    print(f"Evaluating model with arguments: {args}")


def predict(args):
    model_type = args.model_type
    smiles = args.smiles
    
    if model_type == 'model_1':
        prediction = predict_model_1(smiles=smiles)
    elif model_type == 'model_2':
        prediction = predict_model_2(smiles=smiles)
    
    print('P1 :', prediction)
    
    print(f"Predicting with arguments: {args}")

def main():
    parser = argparse.ArgumentParser(description='Servier command-line tools')
    subparsers = parser.add_subparsers()

    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--model_type', type=str, help='Model Type', default='model_1')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
    train_parser.add_argument('--batch_size', type=int, help='Batch Size', default=128)
    train_parser.set_defaults(func=train)

    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--model_type', type=str, help='Model Type', default='model_1')
    eval_parser.add_argument('--epochs', type=str, help='Epochs', default=100)
    eval_parser.add_argument('--batch_size', type=str, help='Batch Size', default=128)
    eval_parser.add_argument('--n_splits', type=str, help='Number of Splits', default=5)
    eval_parser.set_defaults(func=evaluate)

    # Prediction command
    pred_parser = subparsers.add_parser('predict', help='Make predictions')
    pred_parser.add_argument('--model_type', type=str, help='Model Type', default='model_1')
    pred_parser.add_argument('--smiles', type=str, help='SMILES string of the molecule')
    pred_parser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
