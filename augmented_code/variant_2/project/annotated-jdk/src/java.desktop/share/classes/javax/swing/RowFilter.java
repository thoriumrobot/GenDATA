/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2005, 2014, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package javax.swing;

    @Positive
import org.checkerframework.checker.regex.qual.Regex;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.ArrayList;
    @Positive
import java.math.BigDecimal;
    @Positive
import java.math.BigInteger;
    @Positive
import java.util.Date;
    @Positive
import java.util.List;
    @Positive
import java.util.regex.Matcher;
    @Positive
import java.util.regex.Pattern;
    @Positive
import java.util.regex.PatternSyntaxException;

    @Positive
@AnnotatedFor({ "regex" })
    @Positive
public abstract class RowFilter<M, I> {

    @Positive
    public enum ComparisonType {

    @Positive
        BEFORE, AFTER, EQUAL, NOT_EQUAL
    @Positive
    }

    @Positive
    protected RowFilter() {
    @Positive
    }

    @Positive
    public static <M, I> RowFilter<M, I> regexFilter(@Regex String regex, int... indices);

    @Positive
    public static <M, I> RowFilter<M, I> dateFilter(ComparisonType type, Date date, int... indices);

    @Positive
    public static <M, I> RowFilter<M, I> numberFilter(ComparisonType type, Number number, int... indices);

    @Positive
    public static <M, I> RowFilter<M, I> orFilter(Iterable<? extends RowFilter<? super M, ? super I>> filters);

    @Positive
    public static <M, I> RowFilter<M, I> andFilter(Iterable<? extends RowFilter<? super M, ? super I>> filters);

    @Positive
    public static <M, I> RowFilter<M, I> notFilter(RowFilter<M, I> filter);

    @Positive
    public abstract boolean include(Entry<? extends M, ? extends I> entry);

    @Positive
    public abstract static class Entry<M, I> {

    @Positive
        public Entry() {
    @Positive
        }

    @Positive
        public abstract M getModel();

    @Positive
        public abstract int getValueCount();

    @Positive
        public abstract Object getValue(int index);

    @Positive
        public String getStringValue(int index);

    @Positive
        public abstract I getIdentifier();
    @Positive
    }

    @Positive
    private abstract static class GeneralFilter<M, I> extends RowFilter<M, I> {

    @Positive
        @Override
    @Positive
        public boolean include(Entry<? extends M, ? extends I> value);

    @Positive
        protected abstract boolean include(Entry<? extends M, ? extends I> value, int index);
    @Positive
    }

    @Positive
    private static class RegexFilter<M, I> extends GeneralFilter<M, I> {

    @Positive
        @Override
    @Positive
        protected boolean include(Entry<? extends M, ? extends I> value, int index);
    @Positive
    }

    @Positive
    private static class DateFilter<M, I> extends GeneralFilter<M, I> {

    @Positive
        @Override
    @Positive
        protected boolean include(Entry<? extends M, ? extends I> value, int index);
    @Positive
    }

    @Positive
    private static class NumberFilter<M, I> extends GeneralFilter<M, I> {

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        protected boolean include(Entry<? extends M, ? extends I> value, int index);
    @Positive
    }

    @Positive
    private static class OrFilter<M, I> extends RowFilter<M, I> {

    @Positive
        public boolean include(Entry<? extends M, ? extends I> value);
    @Positive
    }

    @Positive
    private static class AndFilter<M, I> extends OrFilter<M, I> {

    @Positive
        public boolean include(Entry<? extends M, ? extends I> value);
    @Positive
    }

    @Positive
    private static class NotFilter<M, I> extends RowFilter<M, I> {

    @Positive
        public boolean include(Entry<? extends M, ? extends I> value);
    @Positive
    }
    @Positive
}
