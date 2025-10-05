/*
 * Copyright (c) 2001, 2020, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.security.cert;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.IOException;
import java.security.PublicKey;
import javax.security.auth.x500.X500Principal;
import sun.security.util.AnchorCertificates;
import sun.security.x509.NameConstraintsExtension;
import sun.security.x509.X500Name;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class TrustAnchor {

    public TrustAnchor(X509Certificate trustedCert, byte[] nameConstraints) {
    }

    public TrustAnchor(X500Principal caPrincipal, PublicKey pubKey, byte[] nameConstraints) {
    }

    public TrustAnchor(String caName, PublicKey pubKey, byte[] nameConstraints) {
    }

    public final X509Certificate getTrustedCert();

    public final X500Principal getCA();

    public final String getCAName();

    public final PublicKey getCAPublicKey();

    public final byte[] getNameConstraints();

    public String toString();

    synchronized boolean isJdkCA();
}
