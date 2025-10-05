/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.xni;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;

    @Positive
public class XNIException extends RuntimeException {

    @Positive
    public XNIException(String message) {
    @Positive
    }

    @Positive
    public XNIException(Exception exception) {
    @Positive
    }

    @Positive
    public XNIException(String message, Exception exception) {
    @Positive
    }

    @Positive
    public Exception getException();

    @Positive
    @Nullable
    @Positive
    public Throwable getCause();
    @Positive
}

// CFWR semantic augmentation - variant 1
