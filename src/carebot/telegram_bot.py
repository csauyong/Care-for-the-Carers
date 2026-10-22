"""Telegram transport. Thin on purpose: all behaviour lives in CareAgent.

Two ways to run:
- polling (default): works anywhere with outbound internet, no public URL needed.
- webhook: set CAREBOT_WEBHOOK_URL to the public https URL of this server;
  suits hosts that sleep idle containers or need an open port.
"""

from __future__ import annotations

import hashlib
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import telebot
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions, Update

from .agent import CareAgent, Reply
from .config import Settings

log = logging.getLogger(__name__)
WEBHOOK_PATH = "/telegram"


def build_bot(token: str, agent: CareAgent) -> telebot.TeleBot:
    # Replies wait on a model call, so handle several chats at once. The agent
    # serialises turns within one chat.
    bot = telebot.TeleBot(token, num_threads=8)

    def send(chat_id: int, reply: Reply) -> None:
        markup = None
        if reply.buttons:
            markup = InlineKeyboardMarkup()
            markup.row(*(InlineKeyboardButton(label, callback_data=data) for label, data in reply.buttons))
        bot.send_message(
            chat_id, reply.text, reply_markup=markup, link_preview_options=LinkPreviewOptions(is_disabled=True)
        )

    @bot.message_handler(commands=["start", "hello"])
    def on_start(message):
        send(message.chat.id, agent.start(str(message.chat.id)))

    @bot.message_handler(commands=["forget"])
    def on_forget(message):
        send(message.chat.id, agent.forget(str(message.chat.id)))

    @bot.message_handler(commands=["help", "resources"])
    def on_resources(message):
        send(message.chat.id, agent.choose(str(message.chat.id), "yes"))

    @bot.message_handler(func=lambda m: True, content_types=["text"])
    def on_text(message):
        bot.send_chat_action(message.chat.id, "typing")
        send(message.chat.id, agent.respond(str(message.chat.id), message.text))

    @bot.callback_query_handler(func=lambda call: True)
    def on_button(call):
        bot.answer_callback_query(call.id)
        chat_id = call.message.chat.id
        # Remove the buttons so an offer can only be answered once.
        bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=None)
        send(chat_id, agent.choose(str(chat_id), call.data))

    return bot


def run_polling(bot: telebot.TeleBot) -> None:
    bot.remove_webhook()
    log.info("Polling Telegram for updates. Press Ctrl+C to stop.")
    bot.infinity_polling(skip_pending=True)


def webhook_secret(token: str) -> str:
    """Stable per-bot secret Telegram echoes back, so we can reject forged requests."""
    return hashlib.sha256(f"carebot-webhook:{token}".encode()).hexdigest()[:48]


def make_webhook_server(bot, secret: str, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._respond(200 if self.path == "/healthz" else 404)

        def do_POST(self):
            if self.path != WEBHOOK_PATH:
                return self._respond(404)
            if self.headers.get("X-Telegram-Bot-Api-Secret-Token") != secret:
                return self._respond(403)
            length = int(self.headers.get("Content-Length") or 0)
            try:
                update = Update.de_json(json.loads(self.rfile.read(length)))
            except ValueError:
                return self._respond(400)
            bot.process_new_updates([update])
            self._respond(200)

        def _respond(self, status: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok" if status == 200 else b"")

        def log_message(self, fmt, *args):
            log.debug("webhook: " + fmt, *args)

    return ThreadingHTTPServer((host, port), Handler)


def run_webhook(bot: telebot.TeleBot, settings: Settings) -> None:
    secret = webhook_secret(settings.telegram_token)
    url = settings.webhook_url.rstrip("/") + WEBHOOK_PATH
    server = make_webhook_server(bot, secret, settings.host, settings.port)
    bot.set_webhook(url=url, secret_token=secret, drop_pending_updates=True)
    log.info("Webhook set to %s; listening on %s:%s (health check: /healthz)", url, settings.host, settings.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def run(settings: Settings, agent: CareAgent) -> None:
    bot = build_bot(settings.telegram_token, agent)
    if settings.webhook_url:
        run_webhook(bot, settings)
    else:
        run_polling(bot)
