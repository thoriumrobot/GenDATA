/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.tools.keytool;

    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
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
import java.io.*;
    @Positive
import java.nio.file.Files;
    @Positive
import java.nio.file.Path;
    @Positive
import java.security.*;
    @Positive
import java.security.cert.Certificate;
    @Positive
import java.security.cert.CertificateFactory;
    @Positive
import java.security.cert.CertStoreException;
    @Positive
import java.security.cert.CRL;
    @Positive
import java.security.cert.X509Certificate;
    @Positive
import java.security.cert.CertificateException;
    @Positive
import java.security.cert.URICertStoreParameters;
    @Positive
import java.security.interfaces.ECKey;
    @Positive
import java.security.interfaces.EdECKey;
    @Positive
import java.security.spec.ECParameterSpec;
    @Positive
import java.text.Collator;
    @Positive
import java.text.MessageFormat;
    @Positive
import java.util.*;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.jar.JarEntry;
    @Positive
import java.util.jar.JarFile;
    @Positive
import java.math.BigInteger;
    @Positive
import java.net.URI;
    @Positive
import java.net.URL;
    @Positive
import java.net.URLClassLoader;
    @Positive
import java.security.cert.CertStore;
    @Positive
import java.security.cert.X509CRL;
    @Positive
import java.security.cert.X509CRLEntry;
    @Positive
import java.security.cert.X509CRLSelector;
    @Positive
import javax.security.auth.x500.X500Principal;
    @Positive
import java.util.Base64;
    @Positive
import sun.security.pkcs12.PKCS12KeyStore;
    @Positive
import sun.security.util.ECKeySizeParameterSpec;
    @Positive
import sun.security.util.KeyUtil;
    @Positive
import sun.security.util.NamedCurve;
    @Positive
import sun.security.util.ObjectIdentifier;
    @Positive
import sun.security.pkcs10.PKCS10;
    @Positive
import sun.security.pkcs10.PKCS10Attribute;
    @Positive
import sun.security.provider.X509Factory;
    @Positive
import sun.security.provider.certpath.ssl.SSLServerCertStore;
    @Positive
import sun.security.util.KnownOIDs;
    @Positive
import sun.security.util.Password;
    @Positive
import sun.security.util.SecurityProperties;
    @Positive
import sun.security.util.SecurityProviderConstants;
    @Positive
import sun.security.util.SignatureUtil;
    @Positive
import javax.crypto.KeyGenerator;
    @Positive
import javax.crypto.SecretKey;
    @Positive
import javax.crypto.SecretKeyFactory;
    @Positive
import javax.crypto.spec.PBEKeySpec;
    @Positive
import sun.security.pkcs.PKCS9Attribute;
    @Positive
import sun.security.tools.KeyStoreUtil;
    @Positive
import sun.security.tools.PathList;
    @Positive
import sun.security.util.DerValue;
    @Positive
import sun.security.util.Pem;
    @Positive
import sun.security.x509.*;
    @Positive
import static java.security.KeyStore.*;
    @Positive
import static sun.security.tools.keytool.Main.Command.*;
    @Positive
import static sun.security.tools.keytool.Main.Option.*;
    @Positive
import sun.security.util.DisabledAlgorithmConstraints;

    @Positive
public final class Main {

    @Positive
    public static void main(String[] args) throws Exception;

    @Positive
    String[] parseArgs(String[] args) throws Exception;

    @Positive
    boolean isKeyStoreRelated(Command cmd);

    @Positive
    void doCommands(PrintStream out) throws Exception;

    @Positive
    boolean inplaceImportCheck() throws Exception;

    @Positive
    KeyStore loadSourceKeyStore() throws Exception;

    @Positive
    public static Collection<? extends CRL> loadCRLs(String src) throws Exception;

    @Positive
    public static List<CRL> readCRLsFromCert(X509Certificate cert) throws Exception;
    @Positive
}

    @Positive
class Pair<A, B> {

    @Positive
    public final A fst;

    @Positive
    public final B snd;

    @Positive
    public Pair(A fst, B snd) {
    @Positive
    }

    @Positive
    public String toString();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object other);

    @Positive
    public int hashCode();

    @Positive
    public static <A, B> Pair<A, B> of(A a, B b);
    @Positive
}

// CFWR semantic augmentation - variant 1
