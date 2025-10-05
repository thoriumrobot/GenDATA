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
package com.sun.org.apache.xml.internal.security.keys.storage;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.security.KeyStore;
    @Positive
import java.security.cert.Certificate;
    @Positive
import java.security.cert.X509Certificate;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.storage.implementations.KeyStoreResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.storage.implementations.SingleCertificateResolver;

    @Positive
public class StorageResolver {

    @Positive
    public StorageResolver(StorageResolverSpi resolver) {
    @Positive
    }

    @Positive
    public StorageResolver(KeyStore keyStore) {
    @Positive
    }

    @Positive
    public StorageResolver(X509Certificate x509certificate) {
    @Positive
    }

    @Positive
    public void add(StorageResolverSpi resolver);

    @Positive
    public void add(KeyStore keyStore);

    @Positive
    public void add(X509Certificate x509certificate);

    @Positive
    public Iterator<Certificate> getIterator();

    @Positive
    static class StorageResolverIterator implements Iterator<Certificate> {

    @Positive
        public StorageResolverIterator(Iterator<StorageResolverSpi> resolvers) {
    @Positive
        }

    @Positive
        @Pure
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public Certificate next();

    @Positive
        public void remove();
    @Positive
    }
    @Positive
}
