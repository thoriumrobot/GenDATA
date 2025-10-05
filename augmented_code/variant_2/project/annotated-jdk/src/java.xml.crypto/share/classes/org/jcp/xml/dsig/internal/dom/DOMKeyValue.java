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
import java.io.IOException;
    @Positive
import java.math.BigInteger;
    @Positive
import java.security.KeyException;
    @Positive
import java.security.KeyFactory;
    @Positive
import java.security.NoSuchAlgorithmException;
    @Positive
import java.security.PublicKey;
    @Positive
import java.security.interfaces.DSAParams;
    @Positive
import java.security.interfaces.DSAPublicKey;
    @Positive
import java.security.interfaces.ECPublicKey;
    @Positive
import java.security.interfaces.RSAPublicKey;
    @Positive
import java.security.spec.DSAPublicKeySpec;
    @Positive
import java.security.spec.ECField;
    @Positive
import java.security.spec.ECFieldFp;
    @Positive
import java.security.spec.ECParameterSpec;
    @Positive
import java.security.spec.ECPoint;
    @Positive
import java.security.spec.ECPublicKeySpec;
    @Positive
import java.security.spec.EllipticCurve;
    @Positive
import java.security.spec.InvalidKeySpecException;
    @Positive
import java.security.spec.KeySpec;
    @Positive
import java.security.spec.RSAPublicKeySpec;
    @Positive
import java.util.Arrays;
    @Positive
import javax.xml.crypto.MarshalException;
    @Positive
import javax.xml.crypto.dom.DOMCryptoContext;
    @Positive
import javax.xml.crypto.dsig.XMLSignature;
    @Positive
import javax.xml.crypto.dsig.keyinfo.KeyValue;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;

    @Positive
public abstract class DOMKeyValue<K extends PublicKey> extends DOMStructure implements KeyValue {

    @Positive
    public DOMKeyValue(K key) throws KeyException {
    @Positive
    }

    @Positive
    public DOMKeyValue(Element kvtElem) throws MarshalException {
    @Positive
    }

    @Positive
    static KeyValue unmarshal(Element kvElem) throws MarshalException;

    @Positive
    public PublicKey getPublicKey() throws KeyException;

    @Positive
    @Override
    @Positive
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
    abstract void marshalPublicKey(Node parent, Document doc, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
    abstract K unmarshalKeyValue(Element kvtElem) throws MarshalException;

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public static BigInteger decode(Element elem) throws MarshalException;

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    static final class RSA extends DOMKeyValue<RSAPublicKey> {

    @Positive
        void marshalPublicKey(Node parent, Document doc, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
        @Override
    @Positive
        RSAPublicKey unmarshalKeyValue(Element kvtElem) throws MarshalException;
    @Positive
    }

    @Positive
    static final class DSA extends DOMKeyValue<DSAPublicKey> {

    @Positive
        @Override
    @Positive
        void marshalPublicKey(Node parent, Document doc, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
        @Override
    @Positive
        DSAPublicKey unmarshalKeyValue(Element kvtElem) throws MarshalException;
    @Positive
    }

    @Positive
    static final class EC extends DOMKeyValue<ECPublicKey> {

    @Positive
        @Override
    @Positive
        void marshalPublicKey(Node parent, Document doc, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
        @Override
    @Positive
        ECPublicKey unmarshalKeyValue(Element kvtElem) throws MarshalException;

    @Positive
        static final class Curve extends ECParameterSpec {
    @Positive
        }
    @Positive
    }

    @Positive
    static final class Unknown extends DOMKeyValue<PublicKey> {

    @Positive
        @Override
    @Positive
        PublicKey unmarshalKeyValue(Element kvElem) throws MarshalException;

    @Positive
        @Override
    @Positive
        void marshalPublicKey(Node parent, Document doc, String dsPrefix, DOMCryptoContext context) throws MarshalException;
    @Positive
    }
    @Positive
}
