#include <string>
#include <sstream>
#include <vector>

#include <string>
#include <cstdarg>
#include <vector>
#include <string>

std::string
vformat (const char *fmt, va_list ap)
{
    // Allocate a buffer on the stack that's big enough for us almost
    // all the time.
    size_t size = 1024;
    char buf[size];

    // Try to vsnprintf into our buffer.
    va_list apcopy;
    va_copy (apcopy, ap);
    int needed = vsnprintf (&buf[0], size, fmt, ap);
    // NB. On Windows, vsnprintf returns -1 if the string didn't fit the
    // buffer.  On Linux & OSX, it returns the length it would have needed.

    if (needed <= size && needed >= 0) {
        // It fit fine the first time, we're done.
        return std::string (&buf[0]);
    } else {
        // vsnprintf reported that it wanted to write more characters
        // than we allotted.  So do a malloc of the right size and try again.
        // This doesn't happen very often if we chose our initial size
        // well.
        std::vector <char> buf;
        size = needed;
        buf.resize (size);
        needed = vsnprintf (&buf[0], size, fmt, apcopy);
        return std::string (&buf[0]);
    }
}

std::string
format (const char *fmt, ...)
{
    va_list ap;
    va_start (ap, fmt);
    std::string buf = vformat (fmt, ap);
    va_end (ap);
    return buf;
}

class Table {
private:
    std::vector<std::vector<std::string>> rows_;

    size_t NumRows() const {
        return rows_.size();
    }
    size_t NumCols() const {
        size_t num_cols = 0;
        for (const auto &row : rows_) {
            num_cols = std::max(row.size(), num_cols);
        }
        return num_cols;
    }

public:
    void Cell(const std::string &s) {
        if (rows_.empty()) {
            NewRow();
        }
        rows_[rows_.size() - 1].push_back(s);
    }

    void Cellf(const char *format, ...) {
        va_list ap;
        va_start (ap, format);
        std::string buf = vformat(format, ap);
        va_end (ap);
        Cell(buf);
    }

    void NewRow() {
        rows_.push_back(std::vector<std::string>());
    }

    std::string csv_str() {
        std::stringstream ss;

        const auto num_cols = NumCols();

        for (const auto row : rows_) {
            for (size_t i = 0; i < num_cols; ++i) {
                if (i < row.size()) {
                    ss << row[i];
                }
                if (i + 1 < num_cols) {
                    ss << ",";
                }
            }
            ss << std::endl;
        }
        return ss.str();
    }
    std::string md_str() {
        return "";
    }
};