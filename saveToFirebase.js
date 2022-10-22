function saveToFirebase(entry) {
    var entryObject = {
        age: age
    };

    firebase.database().ref('entries').push().set(entryObject)
        .then(function(snapshot) {
            success(); // some success method
        }, function(error) {
            console.log('error' + error);
            error(); // some error method
        });
}

saveToFirebase(entry);