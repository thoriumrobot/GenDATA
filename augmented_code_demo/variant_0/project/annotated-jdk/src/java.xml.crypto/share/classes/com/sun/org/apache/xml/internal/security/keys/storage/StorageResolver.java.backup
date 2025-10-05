/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xml.internal.security.keys.storage;

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import java.security.KeyStore;
import java.security.cert.Certificate;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import com.sun.org.apache.xml.internal.security.keys.storage.implementations.KeyStoreResolver;
import com.sun.org.apache.xml.internal.security.keys.storage.implementations.SingleCertificateResolver;

public class StorageResolver {

    public StorageResolver(StorageResolverSpi resolver) {
    }

    public StorageResolver(KeyStore keyStore) {
    }

    public StorageResolver(X509Certificate x509certificate) {
    }

    public void add(StorageResolverSpi resolver);

    public void add(KeyStore keyStore);

    public void add(X509Certificate x509certificate);

    public Iterator<Certificate> getIterator();

    static class StorageResolverIterator implements Iterator<Certificate> {

        public StorageResolverIterator(Iterator<StorageResolverSpi> resolvers) {
        }

        @Pure
        public boolean hasNext();

        @SideEffectsOnly("this")
        public Certificate next();

        public void remove();
    }
}
