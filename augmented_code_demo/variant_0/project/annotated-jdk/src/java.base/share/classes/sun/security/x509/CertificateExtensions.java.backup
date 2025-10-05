/*
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.x509;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.security.cert.CertificateException;
import java.util.*;
import sun.security.util.HexDumpEncoder;
import sun.security.util.*;

public class CertificateExtensions implements CertAttrSet<Extension> {

    public static final String IDENT;

    public static final String NAME;

    public CertificateExtensions() {
    }

    public CertificateExtensions(DerInputStream in) throws IOException {
    }

    public void encode(OutputStream out) throws CertificateException, IOException;

    public void encode(OutputStream out, boolean isCertReq) throws CertificateException, IOException;

    public void set(String name, Object obj) throws IOException;

    public Extension get(String name) throws IOException;

    Extension getExtension(String name);

    public void delete(String name) throws IOException;

    public String getNameByOid(ObjectIdentifier oid) throws IOException;

    public Enumeration<Extension> getElements();

    public Collection<Extension> getAllExtensions();

    public Map<String, Extension> getUnparseableExtensions();

    public String getName();

    public boolean hasUnsupportedCriticalExtension();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object other);

    public int hashCode();

    public String toString();
}

class UnparseableExtension extends Extension {

    public UnparseableExtension(Extension ext, Throwable why) {
    }

    @Override
    public String toString();
}
