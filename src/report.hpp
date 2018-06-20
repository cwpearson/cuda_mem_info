#ifndef REPORT_HPP
#define REPORT_HPP

#include <string>
#include <sstream>
#include <vector>
#include <memory>

#include "element.hpp"

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
    std::vector<std::shared_ptr<Element>> elements_;
public:
    Section(const std::string &title) : title_(title) {}

    std::string ascii_str() const {
        std::stringstream ss;

        ss << title_ << std::endl << "-" << std::endl;

        for (const auto &element : elements_) {
            ss << element->ascii_str() << std::endl;
        }

        return ss.str();
    }

    std::string md_str() const {
        std::stringstream ss;

        ss << "## " << title_ << std::endl;

        for (const auto &element : elements_) {
            ss << element->md_str() << std::endl;
        }

        return ss.str();
    }

    void AppendElement(const std::shared_ptr<Element> &e) {
        elements_.push_back(e);
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

    std::string csv_str() const {
        std::stringstream ss;

        assert(0);

        return ss.str();
    }

    std::string md_str() const {
        std::stringstream ss;

        ss << "# " << title_ <<  std::endl;

        for (const auto &section : sections_) {
            ss << section.md_str() << std::endl;
        }

        return ss.str();
    }

    void AppendSection(const Section &section) {
        sections_.push_back(section);
    }

};



#endif