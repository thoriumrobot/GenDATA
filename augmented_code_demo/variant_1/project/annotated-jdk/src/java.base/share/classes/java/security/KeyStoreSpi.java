/*
    @Positive
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.security;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.*;
    @Positive
import java.util.*;
    @Positive
import java.security.KeyStore.*;
    @Positive
import java.security.cert.Certificate;
    @Positive
import java.security.cert.CertificateException;
    @Positive
import javax.crypto.SecretKey;
    @Positive
import javax.security.auth.callback.*;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class KeyStoreSpi {

    @Positive
    public KeyStoreSpi() {
    @Positive
    }

    @Positive
    public abstract Key engineGetKey(String alias, char[] password) throws NoSuchAlgorithmException, UnrecoverableKeyException;

    @Positive
    public abstract Certificate[] engineGetCertificateChain(String alias);

    @Positive
    public abstract Certificate engineGetCertificate(String alias);

    @Positive
    public abstract Date engineGetCreationDate(String alias);

    @Positive
    public abstract void engineSetKeyEntry(String alias, Key key, char[] password, Certificate[] chain) throws KeyStoreException;

    @Positive
    public abstract void engineSetKeyEntry(String alias, byte[] key, Certificate[] chain) throws KeyStoreException;

    @Positive
    public abstract void engineSetCertificateEntry(String alias, Certificate cert) throws KeyStoreException;

    @Positive
    public abstract void engineDeleteEntry(String alias) throws KeyStoreException;

    @Positive
    public abstract Enumeration<String> engineAliases();

    @Positive
    public abstract boolean engineContainsAlias(String alias);

    @Positive
    public abstract int engineSize();

    @Positive
    public abstract boolean engineIsKeyEntry(String alias);

    @Positive
    public abstract boolean engineIsCertificateEntry(String alias);

    @Positive
    public abstract String engineGetCertificateAlias(Certificate cert);

    @Positive
    public abstract void engineStore(OutputStream stream, char[] password) throws IOException, NoSuchAlgorithmException, CertificateException;

    @Positive
    public void engineStore(KeyStore.LoadStoreParameter param) throws IOException, NoSuchAlgorithmException, CertificateException;

    @Positive
    public abstract void engineLoad(InputStream stream, char[] password) throws IOException, NoSuchAlgorithmException, CertificateException;

    @Positive
    public void engineLoad(KeyStore.LoadStoreParameter param) throws IOException, NoSuchAlgorithmException, CertificateException;

    @Positive
    void engineLoad(InputStream stream, KeyStore.LoadStoreParameter param) throws IOException, NoSuchAlgorithmException, CertificateException;

    @Positive
    public KeyStore.Entry engineGetEntry(String alias, KeyStore.ProtectionParameter protParam) throws KeyStoreException, NoSuchAlgorithmException, UnrecoverableEntryException;

    @Positive
    public void engineSetEntry(String alias, KeyStore.Entry entry, KeyStore.ProtectionParameter protParam) throws KeyStoreException;

    @Positive
    public boolean engineEntryInstanceOf(String alias, Class<? extends KeyStore.Entry> entryClass);

    @Positive
    public boolean engineProbe(InputStream stream) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 1
