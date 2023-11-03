$(function () {
    $('#file-upload').submit(function (e) {
        e.preventDefault();
        var formData = new FormData(this);
        $.ajax({
            url: '{% url "upload" %}',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            xhr: function () {
                var xhr = new window.XMLHttpRequest();
                xhr.upload.addEventListener("progress", function (evt) {
                    if (evt.lengthComputable) {
                        var percentComplete = (evt.loaded / evt.total) * 100;
                        $('#upload-progress').attr('value', percentComplete);
                    }
                }, false);
                return xhr;
            },
            success: function (data) {
                $('#file-details').html(data);
            }
        });
    });

    $('#show-replace-form').click(function () {
        $('#replace-file-form').show();
    });
});