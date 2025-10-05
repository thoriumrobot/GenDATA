/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.util;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signedness.qual.SignedPositive;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import sun.nio.cs.UTF_32BE;
import sun.util.calendar.CalendarDate;
import sun.util.calendar.CalendarSystem;
import java.io.*;
import java.math.BigInteger;
import java.nio.charset.Charset;
import java.util.*;
import static java.nio.charset.StandardCharsets.*;

public class DerValue {

    @SignedPositive
    public static final byte TAG_UNIVERSAL;

    @SignedPositive
    public static final byte TAG_APPLICATION;

    @SignedPositive
    public static final byte TAG_CONTEXT;

    @SignedPositive
    public static final byte TAG_PRIVATE;

    @SignedPositive
    public static final byte tag_Boolean;

    @SignedPositive
    public static final byte tag_Integer;

    @SignedPositive
    public static final byte tag_BitString;

    @SignedPositive
    public static final byte tag_OctetString;

    @SignedPositive
    public static final byte tag_Null;

    @SignedPositive
    public static final byte tag_ObjectId;

    @SignedPositive
    public static final byte tag_Enumerated;

    @SignedPositive
    public static final byte tag_UTF8String;

    @SignedPositive
    public static final byte tag_PrintableString;

    @SignedPositive
    public static final byte tag_T61String;

    @SignedPositive
    public static final byte tag_IA5String;

    @SignedPositive
    public static final byte tag_UtcTime;

    @SignedPositive
    public static final byte tag_GeneralizedTime;

    @SignedPositive
    public static final byte tag_GeneralString;

    @SignedPositive
    public static final byte tag_UniversalString;

    @SignedPositive
    public static final byte tag_BMPString;

    @SignedPositive
    public static final byte tag_Sequence;

    @SignedPositive
    public static final byte tag_SequenceOf;

    @SignedPositive
    public static final byte tag_Set;

    @SignedPositive
    public static final byte tag_SetOf;

    public byte tag;

    public final DerInputStream data;

    public boolean isUniversal();

    public boolean isApplication();

    public boolean isContextSpecific();

    public boolean isContextSpecific(byte cntxtTag);

    boolean isPrivate();

    public boolean isConstructed();

    public boolean isConstructed(byte constructedTag);

    public DerValue(String value) {
    }

    public DerValue(byte stringTag, String value) {
    }

    public DerValue(byte tag, byte[] buffer) {
    }

    public static DerValue wrap(byte tag, DerOutputStream out);

    public DerValue(byte[] encoding) throws IOException {
    }

    public DerValue(InputStream in) throws IOException {
    }

    public void encode(DerOutputStream out) throws IOException;

    public final DerInputStream data();

    public final DerInputStream getData();

    public final byte getTag();

    public boolean getBoolean() throws IOException;

    public ObjectIdentifier getOID() throws IOException;

    public byte[] getOctetString() throws IOException;

    public int getInteger() throws IOException;

    public BigInteger getBigInteger() throws IOException;

    public BigInteger getPositiveBigInteger() throws IOException;

    public int getEnumerated() throws IOException;

    public byte[] getBitString() throws IOException;

    public BitArray getUnalignedBitString() throws IOException;

    public String getAsString() throws IOException;

    public byte[] getBitString(boolean tagImplicit) throws IOException;

    public BitArray getUnalignedBitString(boolean tagImplicit) throws IOException;

    public byte[] getDataBytes() throws IOException;

    public String getPrintableString() throws IOException;

    public String getT61String() throws IOException;

    public String getIA5String() throws IOException;

    public String getBMPString() throws IOException;

    public String getUTF8String() throws IOException;

    public String getGeneralString() throws IOException;

    public String getUniversalString() throws IOException;

    public void getNull() throws IOException;

    public Date getUTCTime() throws IOException;

    public Date getGeneralizedTime() throws IOException;

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public String toString();

    public byte[] toByteArray() throws IOException;

    public DerInputStream toDerInputStream() throws IOException;

    public int length();

    public static boolean isPrintableStringChar(char ch);

    public static byte createTag(byte tagClass, boolean form, byte val);

    public void resetTag(byte tag);

    public DerValue withTag(byte newTag);

    @Override
    public int hashCode();

    DerValue[] subs(byte expectedTag, int startLen) throws IOException;

    public void clear();
}
