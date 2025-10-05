/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xml.internal.security.keys.storage.implementations;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.security.KeyStore;
    @Positive
import java.security.KeyStoreException;
    @Positive
import java.security.cert.Certificate;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.storage.StorageResolverException;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.storage.StorageResolverSpi;

    @Positive
public class KeyStoreResolver extends StorageResolverSpi {

    @Positive
    public KeyStoreResolver(KeyStore keyStore) throws StorageResolverException {
    @Positive
    }

    @Positive
    public Iterator<Certificate> getIterator();

    @Positive
    static class KeyStoreIterator implements Iterator<Certificate> {

    @Positive
        public KeyStoreIterator(KeyStore keyStore) {
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

// CFWR semantic augmentation - variant 0
