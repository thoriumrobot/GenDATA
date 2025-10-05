/*
    @Positive
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.io.ByteArrayOutputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.Reader;
    @Positive
import java.text.Normalizer;
    @Positive
import java.util.*;
    @Positive
import static java.nio.charset.StandardCharsets.UTF_8;
    @Positive
import sun.security.action.GetBooleanAction;
    @Positive
import sun.security.util.*;
    @Positive
import sun.security.pkcs.PKCS9Attribute;

    @Positive
public class AVA implements DerEncoder {

    @Positive
    public AVA(ObjectIdentifier type, DerValue val) {
    @Positive
    }

    @Positive
    public ObjectIdentifier getObjectIdentifier();

    @Positive
    public DerValue getDerValue();

    @Positive
    public String getValueString();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public void encode(DerOutputStream out) throws IOException;

    @Positive
    public void derEncode(OutputStream out) throws IOException;

    @Positive
    public String toString();

    @Positive
    public String toRFC1779String();

    @Positive
    public String toRFC1779String(Map<String, String> oidMap);

    @Positive
    public String toRFC2253String();

    @Positive
    public String toRFC2253String(Map<String, String> oidMap);

    @Positive
    public String toRFC2253CanonicalString();

    @Positive
    boolean hasRFC2253Keyword();
    @Positive
}

    @Positive
class AVAKeyword {

    @Positive
    static ObjectIdentifier getOID(String keyword, int standard, Map<String, String> extraKeywordMap) throws IOException;

    @Positive
    static String getKeyword(ObjectIdentifier oid, int standard);

    @Positive
    static String getKeyword(ObjectIdentifier oid, int standard, Map<String, String> extraOidMap);

    @Positive
    static boolean hasKeyword(ObjectIdentifier oid, int standard);
    @Positive
}

// CFWR semantic augmentation - variant 1
