/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2012, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.util.stream;

    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Collections;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.BinaryOperator;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.Supplier;

    @Positive
@AnnotatedFor({ "lock", "nullness" })
    @Positive
public interface Collector<T, A, R> {

    @Positive
    Supplier<A> supplier();

    @Positive
    BiConsumer<A, T> accumulator();

    @Positive
    BinaryOperator<A> combiner();

    @Positive
    Function<A, R> finisher();

    @Positive
    Set<Characteristics> characteristics();

    @Positive
    public static <T, R> Collector<T, R, R> of(Supplier<R> supplier, BiConsumer<R, T> accumulator, BinaryOperator<R> combiner, Characteristics... characteristics);

    @Positive
    public static <T, A, R> Collector<T, A, R> of(Supplier<A> supplier, BiConsumer<A, T> accumulator, BinaryOperator<A> combiner, Function<A, R> finisher, Characteristics... characteristics);

    @Positive
    enum Characteristics {

    @Positive
        CONCURRENT, UNORDERED, IDENTITY_FINISH
    @Positive
    }
    @Positive
}
