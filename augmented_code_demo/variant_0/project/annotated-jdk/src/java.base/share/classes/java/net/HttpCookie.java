/*
    @Positive
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.net;

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
import java.util.List;
    @Positive
import java.util.StringTokenizer;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.text.SimpleDateFormat;
    @Positive
import java.util.TimeZone;
    @Positive
import java.util.Calendar;
    @Positive
import java.util.GregorianCalendar;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Objects;
    @Positive
import jdk.internal.access.JavaNetHttpCookieAccess;
    @Positive
import jdk.internal.access.SharedSecrets;

    @Positive
public final class HttpCookie implements Cloneable {

    @Positive
    public HttpCookie(String name, String value) {
    @Positive
    }

    @Positive
    public static List<HttpCookie> parse(String header);

    @Positive
    public boolean hasExpired();

    @Positive
    public void setComment(String purpose);

    @Positive
    public String getComment();

    @Positive
    public void setCommentURL(String purpose);

    @Positive
    public String getCommentURL();

    @Positive
    public void setDiscard(boolean discard);

    @Positive
    public boolean getDiscard();

    @Positive
    public void setPortlist(String ports);

    @Positive
    public String getPortlist();

    @Positive
    public void setDomain(String pattern);

    @Positive
    public String getDomain();

    @Positive
    public void setMaxAge(long expiry);

    @Positive
    public long getMaxAge();

    @Positive
    public void setPath(String uri);

    @Positive
    public String getPath();

    @Positive
    public void setSecure(boolean flag);

    @Positive
    public boolean getSecure();

    @Positive
    public String getName();

    @Positive
    public void setValue(String newValue);

    @Positive
    public String getValue();

    @Positive
    public int getVersion();

    @Positive
    public void setVersion(int v);

    @Positive
    public boolean isHttpOnly();

    @Positive
    public void setHttpOnly(boolean httpOnly);

    @Positive
    public static boolean domainMatches(String domain, String host);

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public Object clone();

    @Positive
    long getCreationTime();

    @Positive
    static interface CookieAttributeAssignor {

    @Positive
        public void assign(HttpCookie cookie, String attrName, String attrValue);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
