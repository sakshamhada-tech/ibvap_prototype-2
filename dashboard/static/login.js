const error = new URLSearchParams(window.location.search).get("error");
const message = document.getElementById("login-error");
if (error === "rate") {
  message.textContent = "Too many attempts. Wait one minute and try again.";
  message.hidden = false;
} else if (error === "credentials") {
  message.textContent = "The username or password was not accepted.";
  message.hidden = false;
}
