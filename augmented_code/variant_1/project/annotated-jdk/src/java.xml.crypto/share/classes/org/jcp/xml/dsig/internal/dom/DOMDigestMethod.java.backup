/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package org.jcp.xml.dsig.internal.dom;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import javax.xml.crypto.*;
import javax.xml.crypto.dom.DOMCryptoContext;
import javax.xml.crypto.dsig.*;
import javax.xml.crypto.dsig.spec.DigestMethodParameterSpec;
import java.security.InvalidAlgorithmParameterException;
import java.security.spec.AlgorithmParameterSpec;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

public abstract class DOMDigestMethod extends DOMStructure implements DigestMethod {

    static DigestMethod unmarshal(Element dmElem) throws MarshalException;

    void checkParams(DigestMethodParameterSpec params) throws InvalidAlgorithmParameterException;

    public final AlgorithmParameterSpec getParameterSpec();

    DigestMethodParameterSpec unmarshalParams(Element paramsElem) throws MarshalException;

    @Override
    public void marshal(Node parent, String prefix, DOMCryptoContext context) throws MarshalException;

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public int hashCode();

    void marshalParams(Element parent, String prefix) throws MarshalException;

    abstract String getMessageDigestAlgorithm();

    static final class SHA1 extends DOMDigestMethod {

        public String getAlgorithm();

        String getMessageDigestAlgorithm();
    }

    static final class SHA224 extends DOMDigestMethod {

        @Override
        public String getAlgorithm();

        @Override
        String getMessageDigestAlgorithm();
    }

    static final class SHA256 extends DOMDigestMethod {

        public String getAlgorithm();

        String getMessageDigestAlgorithm();
    }

    static final class SHA384 extends DOMDigestMethod {

        public String getAlgorithm();

        String getMessageDigestAlgorithm();
    }

    static final class SHA512 extends DOMDigestMethod {

        public String getAlgorithm();

        String getMessageDigestAlgorithm();
    }

    static final class RIPEMD160 extends DOMDigestMethod {

        @Override
        public String getAlgorithm();

        @Override
        String getMessageDigestAlgorithm();
    }

    static final class WHIRLPOOL extends DOMDigestMethod {

        @Override
        public String getAlgorithm();

        @Override
        String getMessageDigestAlgorithm();
    }

    static final class SHA3_224 extends DOMDigestMethod {

        @Override
        public String getAlgorithm();

        @Override
        String getMessageDigestAlgorithm();
    }

    static final class SHA3_256 extends DOMDigestMethod {

        @Override
        public String getAlgorithm();

        @Override
        String getMessageDigestAlgorithm();
    }

    static final class SHA3_384 extends DOMDigestMethod {

        @Override
        public String getAlgorithm();

        @Override
        String getMessageDigestAlgorithm();
    }

    static final class SHA3_512 extends DOMDigestMethod {

        @Override
        public String getAlgorithm();

        @Override
        String getMessageDigestAlgorithm();
    }
}
