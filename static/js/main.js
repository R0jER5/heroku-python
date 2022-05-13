$('#button_start').click(function () {
        // var form_data = new FormData($('#-file')[0]);




$.ajax({
            type: 'POST',
            url: '/predict',
            data: null,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data);
                console.log('Success!');
            },
        });
    })