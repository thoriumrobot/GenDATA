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
package java.text;

import org.checkerframework.checker.i18nformatter.qual.I18nFormatFor;
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.InvalidObjectException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Locale;

@AnnotatedFor({ "i18nformatter", "nullness" })
public class MessageFormat extends Format {

    public MessageFormat(String pattern) {
    }

    public MessageFormat(String pattern, Locale locale) {
    }

    public void setLocale(Locale locale);

    public Locale getLocale();

    @SuppressWarnings("fallthrough")
    public void applyPattern(String pattern);

    public String toPattern();

    public void setFormatsByArgumentIndex(Format[] newFormats);

    public void setFormats(Format[] newFormats);

    public void setFormatByArgumentIndex(int argumentIndex, Format newFormat);

    public void setFormat(int formatElementIndex, Format newFormat);

    @Nullable
    public Format[] getFormatsByArgumentIndex();

    public Format[] getFormats();

    public final StringBuffer format(@Nullable Object @Nullable [] arguments, StringBuffer result, @Nullable FieldPosition pos);

    public static String format(@I18nFormatFor("#2") String pattern, @Nullable Object... arguments);

    public final StringBuffer format(Object arguments, StringBuffer result, FieldPosition pos);

    public AttributedCharacterIterator formatToCharacterIterator(Object arguments);

    public Object[] parse(@Nullable String source, ParsePosition pos);

    public Object[] parse(String source) throws ParseException;

    @Nullable
    public Object parseObject(String source, ParsePosition pos);

    public Object clone();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public static class Field extends Format.Field {

        protected Field(String name) {
        }

        @java.io.Serial
        protected Object readResolve() throws InvalidObjectException;

        public static final Field ARGUMENT;
    }
}
