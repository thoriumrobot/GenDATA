
    @DotSeparatedIdentifiers String s4a = Boolean.class.getName();

    // :: error: (assignment)
    @PrimitiveType String s4b = Boolean.class.getName();

    // :: error: (assignment)
    @DotSeparatedIdentifiers String s12 = Nested.class.getName();

    // :: error: (assignment)
    @DotSeparatedIdentifiers String s13 = Inner.class.getName();

    // Primitive types

    @PrimitiveType String prim1 = int.class.getName();

    // :: error: (assignment)
    @DotSeparatedIdentifiers String prim2 = int.class.getName();

    @PrimitiveType String prim3 = boolean.class.getName();

