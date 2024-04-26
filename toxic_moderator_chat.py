from telegram.ext import Updater, MessageHandler, CallbackContext, Filters
from telegram import Update
from model_wrapper_loader import MultiLabelDetectionModel, load_model


model_wrapper = load_model()


def get_full_name(message):
    first_name = message.from_user.first_name
    first_name = first_name[0].upper() + first_name[1:]
    last_name = message.from_user.last_name
    last_name = last_name[0].upper() + last_name[1:]

    return first_name + ' ' + last_name


# Define a function to handle the messages that the bot receives
def message_handler(update: Update, context: CallbackContext):
    # Get the message from the update
    message = update.message

    # Check for image or audio
    print("message.photo: ")
    print(message.photo)
    print("message.audio: ")
    print(message.audio)
    if message.photo or message.audio:
        message.delete()
        str_not_allowed_data_type = f"Deleted {message.content_type} message"
        print(str_not_allowed_data_type)
        context.bot.send_message(chat_id=update.message.chat_id,
                                 text=str_not_allowed_data_type)

        return # Skip further processing for this message

    str_prediction = model_wrapper.predict(message.text)
    if str_prediction != "This message was approved":
        full_name = get_full_name(message)
        str_prediction = full_name + ': ' + str_prediction.lower()
        message.delete()
        context.bot.send_message(chat_id=update.message.chat_id,
                                 text=str_prediction)
        print(str_prediction)
        return

    # Print the message to the console
    print(message.text)


def main() -> None:
    updater = Updater("TOKEN", use_context=True)
    updater.dispatcher.add_handler(MessageHandler(Filters.text, message_handler))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
