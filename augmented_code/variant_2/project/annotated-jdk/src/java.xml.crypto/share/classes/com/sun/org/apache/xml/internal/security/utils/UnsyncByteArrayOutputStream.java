/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @reserved * Positive comment block
    @DO * Positive NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xml.internal.security.utils;

    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.OutputStream;

    @Positive
@AnnotatedFor({ "signedness" })
    @Positive
public class UnsyncByteArrayOutputStream extends OutputStream {

    @Positive
    public UnsyncByteArrayOutputStream() {
    @Positive
    }

    @Positive
    public void write(@PolySigned byte[] arg0);

    @Positive
    public void write(@PolySigned byte[] arg0, int arg1, int arg2);

    @Positive
    public void write(int arg0);

    @Positive
    public byte[] toByteArray();

    @Positive
    public void reset();

    @Positive
    public void writeTo(OutputStream out) throws IOException;
    @Positive
}
