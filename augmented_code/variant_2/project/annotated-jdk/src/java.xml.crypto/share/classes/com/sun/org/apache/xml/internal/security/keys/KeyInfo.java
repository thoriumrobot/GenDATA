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
package com.sun.org.apache.xml.internal.security.keys;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.security.PrivateKey;
    @Positive
import java.security.PublicKey;
    @Positive
import java.security.cert.X509Certificate;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import javax.crypto.SecretKey;
    @Positive
import com.sun.org.apache.xml.internal.security.exceptions.XMLSecurityException;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.DEREncodedKeyValue;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.KeyInfoReference;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.KeyName;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.KeyValue;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.MgmtData;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.PGPData;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.RetrievalMethod;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.SPKIData;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.X509Data;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.keyvalues.DSAKeyValue;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.keyvalues.RSAKeyValue;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.keyresolver.KeyResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.keyresolver.KeyResolverException;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.keyresolver.KeyResolverSpi;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.storage.StorageResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.transforms.Transforms;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.Constants;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.ElementProxy;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.SignatureElementProxy;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;
    @Positive
import org.w3c.dom.Attr;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;

    @Positive
public class KeyInfo extends SignatureElementProxy {

    @Positive
    public KeyInfo(Document doc) {
    @Positive
    }

    @Positive
    public KeyInfo(Element element, String baseURI) throws XMLSecurityException {
    @Positive
    }

    @Positive
    public void setSecureValidation(boolean secureValidation);

    @Positive
    public void setId(String id);

    @Positive
    public String getId();

    @Positive
    public void addKeyName(String keynameString);

    @Positive
    public void add(KeyName keyname);

    @Positive
    public void addKeyValue(PublicKey pk);

    @Positive
    public void addKeyValue(Element unknownKeyValueElement);

    @Positive
    public void add(DSAKeyValue dsakeyvalue);

    @Positive
    public void add(RSAKeyValue rsakeyvalue);

    @Positive
    public void add(PublicKey pk);

    @Positive
    public void add(KeyValue keyvalue);

    @Positive
    public void addMgmtData(String mgmtdata);

    @Positive
    public void add(MgmtData mgmtdata);

    @Positive
    public void add(PGPData pgpdata);

    @Positive
    public void addRetrievalMethod(String uri, Transforms transforms, String Type);

    @Positive
    public void add(RetrievalMethod retrievalmethod);

    @Positive
    public void add(SPKIData spkidata);

    @Positive
    public void add(X509Data x509data);

    @Positive
    public void addDEREncodedKeyValue(PublicKey pk) throws XMLSecurityException;

    @Positive
    public void add(DEREncodedKeyValue derEncodedKeyValue);

    @Positive
    public void addKeyInfoReference(String URI) throws XMLSecurityException;

    @Positive
    public void add(KeyInfoReference keyInfoReference);

    @Positive
    public void addUnknownElement(Element element);

    @Positive
    public int lengthKeyName();

    @Positive
    public int lengthKeyValue();

    @Positive
    public int lengthMgmtData();

    @Positive
    public int lengthPGPData();

    @Positive
    public int lengthRetrievalMethod();

    @Positive
    public int lengthSPKIData();

    @Positive
    public int lengthX509Data();

    @Positive
    public int lengthDEREncodedKeyValue();

    @Positive
    public int lengthKeyInfoReference();

    @Positive
    public int lengthUnknownElement();

    @Positive
    public KeyName itemKeyName(int i) throws XMLSecurityException;

    @Positive
    public KeyValue itemKeyValue(int i) throws XMLSecurityException;

    @Positive
    public MgmtData itemMgmtData(int i) throws XMLSecurityException;

    @Positive
    public PGPData itemPGPData(int i) throws XMLSecurityException;

    @Positive
    public RetrievalMethod itemRetrievalMethod(int i) throws XMLSecurityException;

    @Positive
    public SPKIData itemSPKIData(int i) throws XMLSecurityException;

    @Positive
    public X509Data itemX509Data(int i) throws XMLSecurityException;

    @Positive
    public DEREncodedKeyValue itemDEREncodedKeyValue(int i) throws XMLSecurityException;

    @Positive
    public KeyInfoReference itemKeyInfoReference(int i) throws XMLSecurityException;

    @Positive
    public Element itemUnknownElement(int i);

    @Positive
    public boolean isEmpty();

    @Positive
    @Pure
    @Positive
    public boolean containsKeyName();

    @Positive
    @Pure
    @Positive
    public boolean containsKeyValue();

    @Positive
    @Pure
    @Positive
    public boolean containsMgmtData();

    @Positive
    @Pure
    @Positive
    public boolean containsPGPData();

    @Positive
    @Pure
    @Positive
    public boolean containsRetrievalMethod();

    @Positive
    @Pure
    @Positive
    public boolean containsSPKIData();

    @Positive
    @Pure
    @Positive
    public boolean containsUnknownElement();

    @Positive
    @Pure
    @Positive
    public boolean containsX509Data();

    @Positive
    @Pure
    @Positive
    public boolean containsDEREncodedKeyValue();

    @Positive
    @Pure
    @Positive
    public boolean containsKeyInfoReference();

    @Positive
    public PublicKey getPublicKey() throws KeyResolverException;

    @Positive
    PublicKey getPublicKeyFromStaticResolvers() throws KeyResolverException;

    @Positive
    PublicKey getPublicKeyFromInternalResolvers() throws KeyResolverException;

    @Positive
    public X509Certificate getX509Certificate() throws KeyResolverException;

    @Positive
    X509Certificate getX509CertificateFromStaticResolvers() throws KeyResolverException;

    @Positive
    X509Certificate getX509CertificateFromInternalResolvers() throws KeyResolverException;

    @Positive
    public SecretKey getSecretKey() throws KeyResolverException;

    @Positive
    SecretKey getSecretKeyFromStaticResolvers() throws KeyResolverException;

    @Positive
    SecretKey getSecretKeyFromInternalResolvers() throws KeyResolverException;

    @Positive
    public PrivateKey getPrivateKey() throws KeyResolverException;

    @Positive
    PrivateKey getPrivateKeyFromStaticResolvers() throws KeyResolverException;

    @Positive
    PrivateKey getPrivateKeyFromInternalResolvers() throws KeyResolverException;

    @Positive
    public void registerInternalKeyResolver(KeyResolverSpi realKeyResolver);

    @Positive
    int lengthInternalKeyResolver();

    @Positive
    KeyResolverSpi itemInternalKeyResolver(int i);

    @Positive
    public void addStorageResolver(StorageResolver storageResolver);

    @Positive
    public String getBaseLocalName();
    @Positive
}
