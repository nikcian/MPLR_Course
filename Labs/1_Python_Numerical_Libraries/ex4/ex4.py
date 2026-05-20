import sys

class Book:
    def __init__(self, isbn):
        self.isbn = isbn
        self.boughtCopies = 0
        self.boughtPrice = 0.0
        self.soldCopies = 0
        self.soldPrice = 0.0

def month2str(m):
    h={1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    return h[m]

if __name__ == '__main__':
    
    try:
        f= open(sys.argv[1], 'r')
    except:
        print("Error opening the file")
        sys.exit(1)

    monthSoldBooks = {}
    books = {}

    for line in f:
        isbn, transaction, date, copies, price = line.split()
        copies = int(copies)
        price = float(price)

        if isbn not in books:
            books[isbn] = Book(isbn)
        if transaction == 'B':
            books[isbn].boughtCopies += copies
            books[isbn].boughtPrice += price * copies
        elif transaction == 'S':
            books[isbn].soldCopies += copies
            books[isbn].soldPrice += price * copies

            day, month, year = date.split('/')
            month = int(month)
            year = int(year)
            if (year, month) not in monthSoldBooks:
                monthSoldBooks[(year, month)] = 0
            monthSoldBooks[(year, month)] += copies

    f.close()

    print ("Available copies:")
    for isbn in sorted(books):
        if books[isbn].boughtCopies - books[isbn].soldCopies > 0:
            print ('    %s: %d' % (isbn, books[isbn].boughtCopies - books[isbn].soldCopies))

    print ("Sold books per month:")
    for year, month in sorted(monthSoldBooks):
        print('    %s, %s: %d' % (month2str(month), str(year), monthSoldBooks[(year, month)]))
        
    print ("Gain per book:")
    for isbn in sorted(books):
        if books[isbn].soldCopies > 0:
            soldPrice = books[isbn].soldPrice
            soldCopies = books[isbn].soldCopies
            avgBoughtPrice = books[isbn].boughtPrice / books[isbn].boughtCopies
            
            print ("    %s: %.1f (avg %.1f, sold %d)" % (
                isbn,
                soldPrice - avgBoughtPrice * soldCopies,
                soldPrice / soldCopies - avgBoughtPrice,
                soldCopies))
    
    