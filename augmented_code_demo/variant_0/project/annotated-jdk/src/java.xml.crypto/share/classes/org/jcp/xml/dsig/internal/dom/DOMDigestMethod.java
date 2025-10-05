/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package org.jcp.xml.dsig.internal.dom;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import javax.xml.crypto.*;
    @Positive
import javax.xml.crypto.dom.DOMCryptoContext;
    @Positive
import javax.xml.crypto.dsig.*;
    @Positive
import javax.xml.crypto.dsig.spec.DigestMethodParameterSpec;
    @Positive
import java.security.InvalidAlgorithmParameterException;
    @Positive
import java.security.spec.AlgorithmParameterSpec;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;

    @Positive
public abstract class DOMDigestMethod extends DOMStructure implements DigestMethod {

    @Positive
    static DigestMethod unmarshal(Element dmElem) throws MarshalException;

    @Positive
    void checkParams(DigestMethodParameterSpec params) throws InvalidAlgorithmParameterException;

    @Positive
    public final AlgorithmParameterSpec getParameterSpec();

    @Positive
    DigestMethodParameterSpec unmarshalParams(Element paramsElem) throws MarshalException;

    @Positive
    @Override
    @Positive
    public void marshal(Node parent, String prefix, DOMCryptoContext context) throws MarshalException;

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    void marshalParams(Element parent, String prefix) throws MarshalException;

    @Positive
    abstract String getMessageDigestAlgorithm();

    @Positive
    static final class SHA1 extends DOMDigestMethod {

    @Positive
        public String getAlgorithm();

    @Positive
        String getMessageDigestAlgorithm();
    @Positive
    }

    @Positive
    static final class SHA224 extends DOMDigestMethod {

    @Positive
        @Override
    @Positive
        public String getAlgorithm();

    @Positive
        @Override
    @Positive
        String getMessageDigestAlgorithm();
    @Positive
    }

    @Positive
    static final class SHA256 extends DOMDigestMethod {

    @Positive
        public String getAlgorithm();

    @Positive
        String getMessageDigestAlgorithm();
    @Positive
    }

    @Positive
    static final class SHA384 extends DOMDigestMethod {

    @Positive
        public String getAlgorithm();

    @Positive
        String getMessageDigestAlgorithm();
    @Positive
    }

    @Positive
    static final class SHA512 extends DOMDigestMethod {

    @Positive
        public String getAlgorithm();

    @Positive
        String getMessageDigestAlgorithm();
    @Positive
    }

    @Positive
    static final class RIPEMD160 extends DOMDigestMethod {

    @Positive
        @Override
    @Positive
        public String getAlgorithm();

    @Positive
        @Override
    @Positive
        String getMessageDigestAlgorithm();
    @Positive
    }

    @Positive
    static final class WHIRLPOOL extends DOMDigestMethod {

    @Positive
        @Override
    @Positive
        public String getAlgorithm();

    @Positive
        @Override
    @Positive
        String getMessageDigestAlgorithm();
    @Positive
    }

    @Positive
    static final class SHA3_224 extends DOMDigestMethod {

    @Positive
        @Override
    @Positive
        public String getAlgorithm();

    @Positive
        @Override
    @Positive
        String getMessageDigestAlgorithm();
    @Positive
    }

    @Positive
    static final class SHA3_256 extends DOMDigestMethod {

    @Positive
        @Override
    @Positive
        public String getAlgorithm();

    @Positive
        @Override
    @Positive
        String getMessageDigestAlgorithm();
    @Positive
    }

    @Positive
    static final class SHA3_384 extends DOMDigestMethod {

    @Positive
        @Override
    @Positive
        public String getAlgorithm();

    @Positive
        @Override
    @Positive
        String getMessageDigestAlgorithm();
    @Positive
    }

    @Positive
    static final class SHA3_512 extends DOMDigestMethod {

    @Positive
        @Override
    @Positive
        public String getAlgorithm();

    @Positive
        @Override
    @Positive
        String getMessageDigestAlgorithm();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
