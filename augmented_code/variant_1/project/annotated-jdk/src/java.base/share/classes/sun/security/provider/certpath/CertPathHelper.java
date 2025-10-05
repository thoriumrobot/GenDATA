/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2002, 2020, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.provider.certpath;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Date;
    @Positive
import java.util.Set;
    @Positive
import java.security.cert.TrustAnchor;
    @Positive
import java.security.cert.X509CertSelector;
    @Positive
import java.security.cert.X509CRLSelector;
    @Positive
import sun.security.x509.GeneralNameInterface;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class CertPathHelper {

    @Positive
    protected static CertPathHelper instance;

    @Positive
    protected CertPathHelper() {
    @Positive
    }

    @Positive
    protected abstract void implSetPathToNames(X509CertSelector sel, Set<GeneralNameInterface> names);

    @Positive
    protected abstract void implSetDateAndTime(X509CRLSelector sel, Date date, long skew);

    @Positive
    protected abstract boolean implIsJdkCA(TrustAnchor anchor);

    @Positive
    static void setPathToNames(X509CertSelector sel, Set<GeneralNameInterface> names);

    @Positive
    public static void setDateAndTime(X509CRLSelector sel, Date date, long skew);

    @Positive
    public static boolean isJdkCA(TrustAnchor anchor);
    @Positive
}
