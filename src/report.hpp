#ifndef REPORT_HPP
#define REPORT_HPP

#include <string>
#include <sstream>
#include <vector>

class Element {
public:
    virtual std::string md_str() const = 0;
    virtual std::string ascii_str() const = 0;
};

class Text : public Element {
private:
    std::string text_;
public:

    std::string md_str() const {
        return text_;
    }

    std::string ascii_str() const {
        return text_;
    }
};


class Section {
private:
    std::string title_;
    std::vector<Element> elements_;
public:
    Section(const std::string &title) : title_(title) {}

    std::string ascii_str() const {
        std::stringstream ss;

        ss << title_ << std::endl << "-" << std::endl;

        for (const auto &element : elements_) {
            ss << element.ascii_str() << std::endl;
        }

        return ss.str();
    }
};

class Report {
private:
    std::string title_;
    std::vector<Section> sections_;
public:
    Report(const std::string &title): title_(title) {}

    std::string ascii_str() const {
        std::stringstream ss;

        ss << title_ <<  std::endl << "=" << std::endl;

        for (const auto &section : sections_) {
            ss << section.ascii_str() << std::endl;
        }

        return ss.str();
    }
};



#endif