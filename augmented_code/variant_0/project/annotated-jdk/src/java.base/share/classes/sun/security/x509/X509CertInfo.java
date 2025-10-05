/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2014, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.x509;

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
import java.io.OutputStream;
    @Positive
import java.security.cert.*;
    @Positive
import java.util.*;
    @Positive
import sun.security.util.*;
    @Positive
import sun.security.util.HexDumpEncoder;

    @Positive
public class X509CertInfo implements CertAttrSet<String> {

    @Positive
    public static final String IDENT;

    @Positive
    public static final String NAME;

    @Positive
    public static final String DN_NAME;

    @Positive
    public static final String VERSION;

    @Positive
    public static final String SERIAL_NUMBER;

    @Positive
    public static final String ALGORITHM_ID;

    @Positive
    public static final String ISSUER;

    @Positive
    public static final String SUBJECT;

    @Positive
    public static final String VALIDITY;

    @Positive
    public static final String KEY;

    @Positive
    public static final String ISSUER_ID;

    @Positive
    public static final String SUBJECT_ID;

    @Positive
    public static final String EXTENSIONS;

    @Positive
    protected CertificateVersion version;

    @Positive
    protected CertificateSerialNumber serialNum;

    @Positive
    protected CertificateAlgorithmId algId;

    @Positive
    protected X500Name issuer;

    @Positive
    protected X500Name subject;

    @Positive
    protected CertificateValidity interval;

    @Positive
    protected CertificateX509Key pubKey;

    @Positive
    protected UniqueIdentity issuerUniqueId;

    @Positive
    protected UniqueIdentity subjectUniqueId;

    @Positive
    protected CertificateExtensions extensions;

    @Positive
    public X509CertInfo() {
    @Positive
    }

    @Positive
    public X509CertInfo(byte[] cert) throws CertificateParsingException {
    @Positive
    }

    @Positive
    public X509CertInfo(DerValue derVal) throws CertificateParsingException {
    @Positive
    }

    @Positive
    public void encode(OutputStream out) throws CertificateException, IOException;

    @Positive
    public Enumeration<String> getElements();

    @Positive
    public String getName();

    @Positive
    public byte[] getEncodedInfo() throws CertificateEncodingException;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object other);

    @Positive
    public boolean equals(X509CertInfo other);

    @Positive
    public int hashCode();

    @Positive
    public String toString();

    @Positive
    public void set(String name, Object val) throws CertificateException, IOException;

    @Positive
    public void delete(String name) throws CertificateException, IOException;

    @Positive
    public Object get(String name) throws CertificateException, IOException;
    @Positive
}
