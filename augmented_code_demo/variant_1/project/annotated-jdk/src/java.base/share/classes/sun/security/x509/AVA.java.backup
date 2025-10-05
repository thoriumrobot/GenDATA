/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.Reader;
import java.text.Normalizer;
import java.util.*;
import static java.nio.charset.StandardCharsets.UTF_8;
import sun.security.action.GetBooleanAction;
import sun.security.util.*;
import sun.security.pkcs.PKCS9Attribute;

public class AVA implements DerEncoder {

    public AVA(ObjectIdentifier type, DerValue val) {
    }

    public ObjectIdentifier getObjectIdentifier();

    public DerValue getDerValue();

    public String getValueString();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public void encode(DerOutputStream out) throws IOException;

    public void derEncode(OutputStream out) throws IOException;

    public String toString();

    public String toRFC1779String();

    public String toRFC1779String(Map<String, String> oidMap);

    public String toRFC2253String();

    public String toRFC2253String(Map<String, String> oidMap);

    public String toRFC2253CanonicalString();

    boolean hasRFC2253Keyword();
}

class AVAKeyword {

    static ObjectIdentifier getOID(String keyword, int standard, Map<String, String> extraKeywordMap) throws IOException;

    static String getKeyword(ObjectIdentifier oid, int standard);

    static String getKeyword(ObjectIdentifier oid, int standard, Map<String, String> extraOidMap);

    static boolean hasKeyword(ObjectIdentifier oid, int standard);
}
