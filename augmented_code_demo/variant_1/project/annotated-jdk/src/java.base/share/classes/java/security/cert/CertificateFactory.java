/*
    @Positive
 * Copyright (c) 1998, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.security.cert;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.InputStream;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Objects;
    @Positive
import java.security.Provider;
    @Positive
import java.security.Security;
    @Positive
import java.security.NoSuchAlgorithmException;
    @Positive
import java.security.NoSuchProviderException;
    @Positive
import sun.security.jca.*;
    @Positive
import sun.security.jca.GetInstance.Instance;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class CertificateFactory {

    @Positive
    protected CertificateFactory(CertificateFactorySpi certFacSpi, Provider provider, String type) {
    @Positive
    }

    @Positive
    public static final CertificateFactory getInstance(String type) throws CertificateException;

    @Positive
    public static final CertificateFactory getInstance(String type, String provider) throws CertificateException, NoSuchProviderException;

    @Positive
    public static final CertificateFactory getInstance(String type, Provider provider) throws CertificateException;

    @Positive
    public final Provider getProvider();

    @Positive
    public final String getType();

    @Positive
    public final Certificate generateCertificate(InputStream inStream) throws CertificateException;

    @Positive
    public final Iterator<String> getCertPathEncodings();

    @Positive
    public final CertPath generateCertPath(InputStream inStream) throws CertificateException;

    @Positive
    public final CertPath generateCertPath(InputStream inStream, String encoding) throws CertificateException;

    @Positive
    public final CertPath generateCertPath(List<? extends Certificate> certificates) throws CertificateException;

    @Positive
    public final Collection<? extends Certificate> generateCertificates(InputStream inStream) throws CertificateException;

    @Positive
    public final CRL generateCRL(InputStream inStream) throws CRLException;

    @Positive
    public final Collection<? extends CRL> generateCRLs(InputStream inStream) throws CRLException;
    @Positive
}

// CFWR semantic augmentation - variant 1
