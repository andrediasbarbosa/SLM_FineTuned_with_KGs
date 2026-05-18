#pragma once
#include <string>
#include <variant>

namespace slmkg {

struct Error {
    std::string message;
    explicit Error(std::string msg) : message(std::move(msg)) {}
};

template<typename T>
class Result {
public:
    static Result ok(T val) {
        Result r;
        r.storage_ = std::move(val);
        return r;
    }
    static Result err(std::string msg) {
        Result r;
        r.storage_ = Error{std::move(msg)};
        return r;
    }

    bool is_ok()  const { return std::holds_alternative<T>(storage_); }
    bool is_err() const { return std::holds_alternative<Error>(storage_); }

    const T& value() const { return std::get<T>(storage_); }
    T&       value()       { return std::get<T>(storage_); }

    const std::string& error_message() const {
        return std::get<Error>(storage_).message;
    }

private:
    std::variant<T, Error> storage_;
};

template<>
class Result<void> {
public:
    static Result ok()                  { return Result{true, {}}; }
    static Result err(std::string msg)  { return Result{false, std::move(msg)}; }

    bool is_ok()  const { return ok_; }
    bool is_err() const { return !ok_; }

    const std::string& error_message() const { return msg_; }

private:
    Result(bool ok, std::string msg) : ok_(ok), msg_(std::move(msg)) {}
    bool        ok_ = true;
    std::string msg_;
};

} // namespace slmkg
