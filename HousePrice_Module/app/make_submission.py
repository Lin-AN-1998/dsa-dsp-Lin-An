def submission_file(X_test_id, y_pred):
    with open('./submission.csv', 'w') as writer:
        n = len(y_pred)
        
        writer.write('Id,SalePrice')
        writer.write('\n')

        for i in range(n):
            line = str(X_test_id[i]) + ',' + str(y_pred[i])
            writer.write(line)
            writer.write('\n')