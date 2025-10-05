/*
 * Copyright (c) 2005, 2014, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing;

import org.checkerframework.checker.regex.qual.Regex;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.ArrayList;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.Date;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

@AnnotatedFor({ "regex" })
public abstract class RowFilter<M, I> {

    public enum ComparisonType {

        BEFORE, AFTER, EQUAL, NOT_EQUAL
    }

    protected RowFilter() {
    }

    public static <M, I> RowFilter<M, I> regexFilter(@Regex String regex, int... indices);

    public static <M, I> RowFilter<M, I> dateFilter(ComparisonType type, Date date, int... indices);

    public static <M, I> RowFilter<M, I> numberFilter(ComparisonType type, Number number, int... indices);

    public static <M, I> RowFilter<M, I> orFilter(Iterable<? extends RowFilter<? super M, ? super I>> filters);

    public static <M, I> RowFilter<M, I> andFilter(Iterable<? extends RowFilter<? super M, ? super I>> filters);

    public static <M, I> RowFilter<M, I> notFilter(RowFilter<M, I> filter);

    public abstract boolean include(Entry<? extends M, ? extends I> entry);

    public abstract static class Entry<M, I> {

        public Entry() {
        }

        public abstract M getModel();

        public abstract int getValueCount();

        public abstract Object getValue(int index);

        public String getStringValue(int index);

        public abstract I getIdentifier();
    }

    private abstract static class GeneralFilter<M, I> extends RowFilter<M, I> {

        @Override
        public boolean include(Entry<? extends M, ? extends I> value);

        protected abstract boolean include(Entry<? extends M, ? extends I> value, int index);
    }

    private static class RegexFilter<M, I> extends GeneralFilter<M, I> {

        @Override
        protected boolean include(Entry<? extends M, ? extends I> value, int index);
    }

    private static class DateFilter<M, I> extends GeneralFilter<M, I> {

        @Override
        protected boolean include(Entry<? extends M, ? extends I> value, int index);
    }

    private static class NumberFilter<M, I> extends GeneralFilter<M, I> {

        @Override
        @SuppressWarnings("unchecked")
        protected boolean include(Entry<? extends M, ? extends I> value, int index);
    }

    private static class OrFilter<M, I> extends RowFilter<M, I> {

        public boolean include(Entry<? extends M, ? extends I> value);
    }

    private static class AndFilter<M, I> extends OrFilter<M, I> {

        public boolean include(Entry<? extends M, ? extends I> value);
    }

    private static class NotFilter<M, I> extends RowFilter<M, I> {

        public boolean include(Entry<? extends M, ? extends I> value);
    }
}
