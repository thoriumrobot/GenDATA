/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.taglets;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.List;
    @Positive
import java.util.Set;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import com.sun.source.doctree.DocTree;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.RawHtml;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Content;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
    @Positive
import static jdk.javadoc.doclet.Taglet.Location.*;

    @Positive
public final class UserTaglet implements Taglet {

    @Positive
    public UserTaglet(jdk.javadoc.doclet.Taglet t) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public Set<jdk.javadoc.doclet.Taglet.Location> getAllowedLocations();

    @Positive
    @Override
    @Positive
    public boolean inField();

    @Positive
    @Override
    @Positive
    public boolean inConstructor();

    @Positive
    @Override
    @Positive
    public boolean inMethod();

    @Positive
    @Override
    @Positive
    public boolean inOverview();

    @Positive
    @Override
    @Positive
    public boolean inModule();

    @Positive
    @Override
    @Positive
    public boolean inPackage();

    @Positive
    @Override
    @Positive
    public boolean inType();

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public boolean isInlineTag();

    @Positive
    @Override
    @Positive
    public boolean isBlockTag();

    @Positive
    @Override
    @Positive
    public String getName();

    @Positive
    @Override
    @Positive
    public Content getInlineTagOutput(Element element, DocTree tag, TagletWriter writer);

    @Positive
    @Override
    @Positive
    public Content getAllBlockTagOutput(Element holder, TagletWriter writer);
    @Positive
}
