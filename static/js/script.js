document.addEventListener("DOMContentLoaded", function() {
    var socket = io.connect('http://' + document.domain + ':' + location.port);

    socket.on('update', function(msg) {
        console.log("Update received: " + msg.message);
        var statusDiv = document.getElementById('status');
        if (statusDiv) {
            // Met à jour le contenu de la div existante
            statusDiv.innerHTML = '<div class="notification">' + msg.message + '</div>';
        }
    });
});