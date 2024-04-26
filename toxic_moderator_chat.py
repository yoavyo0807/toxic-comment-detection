from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, Filters
from telegram import Update
from model_wrapper_loader import MultiLabelDetectionModel, load_model


model_wrapper = load_model()


# def start(update: Update, context: CallbackContext) -> None:
#     context.bot.send_message(chat_id=update.message.chat_id, text="Hi all, this is a toxic free group chat. Enjoy your time!")


def edit(update: Update, context: CallbackContext, new_text: str) -> None:
    print("***")
    print("edit func")
    print("update:")
    print(update)
    print("context:")
    print(context)
    print("new_text:")
    print(new_text)
    print("***")

    context.bot.editMessageText(chat_id=update.message.chat_id,
                                message_id=update.message.reply_to_message.message_id,
                                text=new_text)


# Define a function to handle the messages that the bot receives
def message_handler(update: Update, context: CallbackContext):
    # Get the message from the update
    print("---")
    print("I am in message_handler")
    print("update: ")
    print(update)
    print("context: ")
    print(context)
    print("---")
    message = update.message
    str_prediction = model_wrapper.predict(message.text)
    if str_prediction != "This message was approved":
        message.text = str_prediction
        edit(update, context, str_prediction)
    
    # Check for image or audio
    if message.photo or message.audio:
        message.delete()
        print(f"Deleted {message.content_type} message")
        return # Skip further processing for this message

    # Print the message to the console
    print(message.text)


def main() -> None:
    updater = Updater("TOKEN", use_context=True)
    # updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('edit', edit))
    updater.dispatcher.add_handler(MessageHandler(Filters.text, message_handler))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
