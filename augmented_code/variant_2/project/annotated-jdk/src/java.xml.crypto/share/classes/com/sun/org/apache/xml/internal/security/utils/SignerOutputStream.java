/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xml.internal.security.utils;

    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.ByteArrayOutputStream;
    @Positive
import com.sun.org.apache.xml.internal.security.algorithms.SignatureAlgorithm;
    @Positive
import com.sun.org.apache.xml.internal.security.signature.XMLSignatureException;

    @Positive
@AnnotatedFor("signedness")
    @Positive
public class SignerOutputStream extends ByteArrayOutputStream {

    @Positive
    public SignerOutputStream(SignatureAlgorithm sa) {
    @Positive
    }

    @Positive
    public void write(byte[] arg0);

    @Positive
    public void write(@PolySigned int arg0);

    @Positive
    public void write(@PolySigned byte[] arg0, int arg1, int arg2);
    @Positive
}
