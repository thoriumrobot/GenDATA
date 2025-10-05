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
import java.security.Key;
    @Positive
import java.security.InvalidAlgorithmParameterException;
    @Positive
import java.security.InvalidKeyException;
    @Positive
import java.security.SignatureException;
    @Positive
import java.security.spec.AlgorithmParameterSpec;
    @Positive
import javax.xml.crypto.MarshalException;
    @Positive
import javax.xml.crypto.dom.DOMCryptoContext;
    @Positive
import javax.xml.crypto.dsig.SignatureMethod;
    @Positive
import javax.xml.crypto.dsig.SignedInfo;
    @Positive
import javax.xml.crypto.dsig.XMLSignature;
    @Positive
import javax.xml.crypto.dsig.XMLSignatureException;
    @Positive
import javax.xml.crypto.dsig.XMLSignContext;
    @Positive
import javax.xml.crypto.dsig.XMLValidateContext;
    @Positive
import javax.xml.crypto.dsig.spec.SignatureMethodParameterSpec;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;

    @Positive
abstract class AbstractDOMSignatureMethod extends DOMStructure implements SignatureMethod {

    @Positive
    abstract boolean verify(Key key, SignedInfo si, byte[] sig, XMLValidateContext context) throws InvalidKeyException, SignatureException, XMLSignatureException;

    @Positive
    abstract byte[] sign(Key key, SignedInfo si, XMLSignContext context) throws InvalidKeyException, XMLSignatureException;

    @Positive
    abstract String getJCAAlgorithm();

    @Positive
    abstract Type getAlgorithmType();

    @Positive
    @Override
    @Positive
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
    void marshalParams(Element parent, String paramsPrefix) throws MarshalException;

    @Positive
    SignatureMethodParameterSpec unmarshalParams(Element paramsElem) throws MarshalException;

    @Positive
    void checkParams(SignatureMethodParameterSpec params) throws InvalidAlgorithmParameterException;

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
    boolean paramsEqual(AlgorithmParameterSpec spec);
    @Positive
}

// CFWR semantic augmentation - variant 0
